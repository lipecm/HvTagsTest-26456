package org.firstinspires.ftc.teamcode;

import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagMetadata;
import org.firstinspires.ftc.vision.apriltag.AprilTagPoseFtc;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;
import org.horizon36596.simloop.field.ObstacleBox;
import org.horizon36596.simloop.vision.AprilTagLayout;
import org.horizon36596.simloop.vision.CameraLens;
import org.horizon36596.simloop.vision.CameraMount;
import org.horizon36596.simloop.vision.SimTagCamera;
import org.horizon36596.simloop.vision.TagSighting;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cross-check against SimLoop's ground truth.
 *
 * <p>{@link AprilTagTargetingTest} builds detections by hand, which proves the rules but not
 * that the numbers feeding them mean anything. Here the geometry comes from SimLoop's
 * simulated tag camera - an independent implementation of where a tag really is - and the
 * sighting is turned into the SDK detection shape the OpMode receives. Range, bearing and
 * elevation are then checked against the geometry, and the targeting rules are driven with
 * the result.
 *
 * <p>SimLoop ships no field and no tag layout: the numbers below are invented, exactly as its
 * documentation asks, and are only used as a place to stand the camera.
 */
class AprilTagSimVisionTest {

    /** Camera: 65 deg horizontal FOV at 640x480, 6 in forward and 10 in up, looking straight ahead. */
    private static SimTagCamera camera() {
        CameraLens lens = CameraLens.fromHorizontalFov(65.0, 640, 480);
        CameraMount mount = CameraMount.front(6.0, 10.0, 0.0);
        return new SimTagCamera(lens, mount, 0.0, Double.POSITIVE_INFINITY, 90.0,
                Collections.<ObstacleBox>emptyList());
    }

    private static AprilTagLayout oneTag(double x, double y, double z) {
        return AprilTagLayout.builder()
                .addTag(1, x, y, z, Math.PI, 5.0)   // id, x, y, z (in), facing (rad), size (in)
                .build();
    }

    /** The distance from the camera's own position to the tag, computed independently. */
    private static double expectedRange(double tagX, double tagY, double tagZ) {
        double dx = tagX - 6.0;
        double dy = tagY - 0.0;
        double dz = tagZ - 10.0;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /** What the OpMode would receive: the sighting reshaped into an SDK single detection. */
    private static AprilTagSingleDetection asSdkDetection(TagSighting sighting, String name) {
        double[][] px = sighting.getCornersPx();
        Point[] corners = new Point[px.length];
        for (int i = 0; i < px.length; i++) {
            corners[i] = new Point(px[i][0], px[i][1]);
        }
        AprilTagPoseFtc pose = new AprilTagPoseFtc(
                0.0, 0.0, 0.0,
                sighting.getTxDegrees(), sighting.getTyDegrees(), 0.0,
                sighting.getRangeInches(), sighting.getTxDegrees(), sighting.getTyDegrees());
        AprilTagMetadata metadata = new AprilTagMetadata(sighting.getId(), name, 5.0,
                DistanceUnit.INCH);
        return new AprilTagSingleDetection(sighting.getId(), 0, 95f,
                new Point(sighting.getCentreXPx(), sighting.getCentreYPx()), corners,
                metadata, pose, null, null, 0L, DistanceUnit.INCH);
    }

    @Test
    void simulatedSightingRangeMatchesTheGeometry() {
        List<TagSighting> seen = camera().observe(oneTag(70.0, 30.0, 6.0), 0.0, 0.0, 0.0);

        assertEquals(1, seen.size(), "one tag in front of the camera is one sighting");
        TagSighting sighting = seen.get(0);
        assertEquals(1, sighting.getId());

        assertEquals(expectedRange(70.0, 30.0, 6.0), sighting.getRangeInches(), 0.05,
                "the simulator's range must agree with the distance it was given");
        assertEquals(Math.toDegrees(Math.atan2(30.0, 64.0)), Math.abs(sighting.getTxDegrees()),
                0.05, "bearing is the horizontal angle off the camera axis");
        assertEquals(Math.toDegrees(Math.atan2(4.0, 64.0)), Math.abs(sighting.getTyDegrees()),
                0.05, "elevation follows the FTC ty convention: atan(vertical / depth)");
        assertTrue(sighting.getAreaPercent() > 0.0, "the tag is inside the image");
    }

    @Test
    void targetingRulesDriveOffTheSimulatedGeometry() {
        TagSighting sighting = camera().observe(oneTag(70.0, 30.0, 6.0), 0.0, 0.0, 0.0).get(0);
        AprilTagSingleDetection detection = asSdkDetection(sighting, "R_HIVE");

        AprilTagConfig config = new AprilTagConfig();
        assertTrue(AprilTagTargeting.isScorable(detection, config),
                "a RED-named target is scorable on a RED alliance");

        config.applyJson("{\"alliance\":\"BLUE\"}");
        assertFalse(AprilTagTargeting.isScorable(detection, config),
                "the same tag is not a target for the other alliance");

        config.applyJson("{\"alliance\":\"RED\"}");
        AprilTagDetection best = AprilTagTargeting.bestTarget(
                Collections.singletonList(detection), config);
        assertSame(detection, best);
        assertEquals(sighting.getRangeInches(), best.ftcPose.range, 1e-6,
                "the range the dashboard shows must be the range the simulator measured");
        assertEquals(sighting.getTxDegrees(), best.ftcPose.bearing, 1e-6);
    }

    @Test
    void theNearestOfTwoSimulatedTagsWins() {
        AprilTagLayout twoTags = AprilTagLayout.builder()
                .addTag(1, 40.0, 0.0, 6.0, Math.PI, 5.0)
                .addTag(2, 75.0, 0.0, 6.0, Math.PI, 5.0)
                .build();

        List<TagSighting> seen = camera().observe(twoTags, 0.0, 0.0, 0.0);
        assertEquals(2, seen.size());

        TagSighting near = null;
        TagSighting far = null;
        for (TagSighting s : seen) {
            if (s.getId() == 1) {
                near = s;
            } else {
                far = s;
            }
        }
        assertNotNull(near);
        assertNotNull(far);
        assertTrue(near.getRangeInches() < far.getRangeInches());

        List<AprilTagDetection> detections = new ArrayList<>();
        detections.add(asSdkDetection(near, "R_NEAR"));
        detections.add(asSdkDetection(far, "R_FAR"));

        AprilTagDetection best =
                AprilTagTargeting.bestTarget(detections, new AprilTagConfig());
        assertNotNull(best);
        assertEquals(1, ((AprilTagSingleDetection) best).id, "the closest scorable tag is the target");
        assertEquals(near.getRangeInches(), best.ftcPose.range, 1e-6);
    }

    @Test
    void anOutOfFrameTagProducesNoSighting() {
        // Far off to the side: outside a 65 degree lens.
        List<TagSighting> seen = camera().observe(oneTag(20.0, 120.0, 6.0), 0.0, 0.0, 0.0);

        assertTrue(seen.isEmpty(), "a tag outside the FOV must not be reported");
        assertTrue(AprilTagTargeting.bestTarget(new ArrayList<>(), new AprilTagConfig()) == null);
    }
}
