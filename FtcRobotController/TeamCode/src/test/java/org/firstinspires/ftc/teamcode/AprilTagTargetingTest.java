package org.firstinspires.ftc.teamcode;

import org.firstinspires.ftc.robotcore.external.matrices.VectorF;
import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import org.firstinspires.ftc.robotcore.external.navigation.Quaternion;
import org.firstinspires.ftc.vision.apriltag.AprilTagClusterDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagClusterMemberMetadata;
import org.firstinspires.ftc.vision.apriltag.AprilTagClusterMetadata;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagMetadata;
import org.firstinspires.ftc.vision.apriltag.AprilTagPoseFtc;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The scorability rules of the FTC AprilTag Clusters tech tip, run against the SDK 12.0
 * detection types that {@code AprilTagProcessor} actually hands the OpMode.
 *
 * <p>Every detection here is constructed directly rather than decoded from pixels, because the
 * question under test is the RULES, not the decoder. {@link AprilTagSimVisionTest} is the other
 * half: geometry coming from SimLoop's simulated camera.
 */
class AprilTagTargetingTest {

    /** Every pose field the rules do not read is zero; only roll and range carry a value. */
    private static AprilTagPoseFtc pose(double rollDeg, double rangeIn) {
        return new AprilTagPoseFtc(0.0, 0.0, rangeIn, 0.0, 0.0, rollDeg, rangeIn, 0.0, 0.0);
    }

    private static AprilTagSingleDetection single(int id, String name, float decisionMargin,
                                                  double rollDeg, double rangeIn) {
        AprilTagMetadata metadata = name == null ? null
                : new AprilTagMetadata(id, name, 5.0, DistanceUnit.INCH);
        Point[] corners = {
                new Point(308, 228), new Point(332, 228),
                new Point(332, 252), new Point(308, 252)
        };
        return new AprilTagSingleDetection(id, 0, decisionMargin, new Point(320, 240), corners,
                metadata, pose(rollDeg, rangeIn), null, null, 0L, DistanceUnit.INCH);
    }

    private static AprilTagClusterDetection cluster(int percentFound, String name,
                                                   double rollDeg, double rangeIn) {
        // The SDK refuses a cluster of fewer than two members, so both are always here.
        ArrayList<AprilTagClusterMemberMetadata> members = new ArrayList<>();
        members.add(new AprilTagClusterMemberMetadata(1, new VectorF(0f, 0f, 0f), 5.0));
        members.add(new AprilTagClusterMemberMetadata(2, new VectorF(6f, 0f, 0f), 5.0));
        AprilTagClusterMetadata metadata = new AprilTagClusterMetadata(
                members, name, name, new VectorF(0f, 0f, 0f), DistanceUnit.INCH, new Quaternion());
        return new AprilTagClusterDetection(percentFound, metadata, DistanceUnit.INCH,
                pose(rollDeg, rangeIn), null, null, 0L);
    }

    @Test
    void matchingAllianceIsScorableAndOppositeIsNot() {
        AprilTagConfig config = new AprilTagConfig();

        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 0.0, 40.0), config),
                "RED alliance must accept a RED-named target");

        config.applyJson("{\"alliance\":\"BLUE\"}");
        assertFalse(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 0.0, 40.0), config),
                "BLUE alliance must reject a RED-named target");
        assertTrue(AprilTagTargeting.isScorable(single(2, "B_HIVE", 95f, 0.0, 40.0), config),
                "BLUE alliance must accept a BLUE-named target");

        config.applyJson("{\"alliance\":\"ANY\"}");
        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 0.0, 40.0), config));
        assertTrue(AprilTagTargeting.isScorable(single(2, "B_HIVE", 95f, 0.0, 40.0), config));
    }

    @Test
    void uprightGateRejectsAnUpsideDownTarget() {
        AprilTagConfig config = new AprilTagConfig();

        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 0.0, 40.0), config));
        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 90.0, 40.0), config),
                "|roll| exactly at the threshold is still upright");
        assertFalse(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 90.1, 40.0), config),
                "just past the threshold means the cell is upside down");
        assertFalse(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, -179.0, 40.0), config),
                "the gate is on |roll|, so a negative roll counts too");

        config.applyJson("{\"maxRollDeg\":45}");
        assertFalse(AprilTagTargeting.isScorable(single(1, "R_HIVE", 95f, 60.0, 40.0), config),
                "a tighter threshold must reject what the default accepted");
    }

    @Test
    void decisionMarginGateAppliesToSingles() {
        AprilTagConfig config = new AprilTagConfig();

        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 80f, 0.0, 40.0), config));

        config.applyJson("{\"minDecisionMargin\":50}");
        assertFalse(AprilTagTargeting.isScorable(single(1, "R_HIVE", 49f, 0.0, 40.0), config),
                "a weak decode must be dropped once a margin floor is set");
        assertTrue(AprilTagTargeting.isScorable(single(1, "R_HIVE", 51f, 0.0, 40.0), config));
    }

    @Test
    void unknownIdWithoutMetadataIsStillScorable() {
        AprilTagConfig config = new AprilTagConfig();

        AprilTagSingleDetection unknown = single(99, null, 80f, 0.0, 40.0);
        assertNull(unknown.metadata, "test setup: this id has no library entry");
        assertTrue(AprilTagTargeting.isScorable(unknown, config),
                "no metadata means no alliance hint to reject on");
    }

    @Test
    void clusterRulesApplyAlliancePercentAndRoll() {
        AprilTagConfig config = new AprilTagConfig();

        assertTrue(AprilTagTargeting.isScorable(cluster(100, "R_CLUSTER", 0.0, 30.0), config));

        config.applyJson("{\"alliance\":\"BLUE\"}");
        assertFalse(AprilTagTargeting.isScorable(cluster(100, "R_CLUSTER", 0.0, 30.0), config));

        config.applyJson("{\"alliance\":\"RED\",\"minClusterPercent\":80}");
        assertFalse(AprilTagTargeting.isScorable(cluster(60, "R_CLUSTER", 0.0, 30.0), config),
                "a half-seen cluster must be dropped by the percent floor");
        assertTrue(AprilTagTargeting.isScorable(cluster(95, "R_CLUSTER", 0.0, 30.0), config));

        config.applyJson("{\"minClusterPercent\":0,\"maxRollDeg\":90}");
        assertFalse(AprilTagTargeting.isScorable(cluster(100, "R_CLUSTER", 150.0, 30.0), config),
                "the upright rule applies to clusters too");
    }

    @Test
    void bestTargetIsTheNearestScorableAndNothingElse() {
        AprilTagConfig config = new AprilTagConfig();

        AprilTagDetection farScorable = single(1, "R_HIVE", 95f, 0.0, 45.0);
        AprilTagDetection nearWrongAlliance = single(2, "B_HIVE", 95f, 0.0, 18.0);
        AprilTagDetection upsideDown = single(3, "R_HIVE", 95f, 160.0, 12.0);
        List<AprilTagDetection> seen = Arrays.asList(farScorable, nearWrongAlliance, upsideDown);

        AprilTagDetection best = AprilTagTargeting.bestTarget(seen, config);
        assertSame(farScorable, best, "distance only decides between targets that are scorable");

        config.applyJson("{\"alliance\":\"BLUE\"}");
        assertNull(AprilTagTargeting.bestTarget(Arrays.asList(farScorable, upsideDown), config),
                "nothing scorable means no target, not the nearest unusable one");

        assertNull(AprilTagTargeting.bestTarget(null, config), "a null list is not a crash");
        assertNull(AprilTagTargeting.bestTarget(Collections.emptyList(), config));
    }

    @Test
    void namesWithoutAnAlliancePrefixAreNeverFiltered() {
        assertTrue(AprilTagTargeting.allianceOk(null, "RED"));
        assertTrue(AprilTagTargeting.allianceOk("", "RED"));
        assertTrue(AprilTagTargeting.allianceOk("HIVE CELL", "RED"),
                "a name with no R/B prefix carries no alliance hint");
        assertTrue(AprilTagTargeting.allianceOk("RED", "ANY"), "ANY accepts everything");
        assertFalse(AprilTagTargeting.allianceOk("BLUEBIRD", "RED"),
                "the first character is the hint, so a B name is a BLUE target");

        // Only the FIRST character is the hint: a name that merely contains an alliance
        // letter later on must not be filtered out.
        assertTrue(AprilTagTargeting.allianceOk("HIVE BLUE", "RED"));
        assertFalse(AprilTagTargeting.allianceOk("red_wall", "BLUE"));
        assertTrue(AprilTagTargeting.allianceOk("red_wall", "RED"));
        assertTrue(AprilTagTargeting.allianceOk("R_HIVE", null), "a null alliance never filters");
    }
}
