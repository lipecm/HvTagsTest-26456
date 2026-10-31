package org.firstinspires.ftc.teamcode;

import com.qualcomm.hardware.limelightvision.LLResult;

import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;
import org.horizon36596.simloop.fakehardware.FakeLimelight3A;
import org.horizon36596.simloop.vision.AprilTagLayout;
import org.horizon36596.simloop.vision.CameraMount;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Limelight -&gt; SDK detection conversion, against geometry rather than fixtures.
 *
 * <p>{@link FakeLimelight3A} is the device: it builds the real {@code /results} JSON from
 * SimLoop's tag camera and hands it to the SDK's own {@code LLResult} parser, so everything
 * from {@code t6t_cs} to {@code tx} here is the same bytecode the robot runs. The bridge's job
 * is only the reshape: metres to inches, {@code tx/ty} to bearing/elevation, one
 * {@code AprilTagSingleDetection} per fiducial.
 *
 * <p>Geometry, camera at (6, 0, 10) in the field frame looking down +X (robot pose 0,0,0, mount
 * 6 in forward, 10 in up, no pitch):
 * <pre>
 *   tag 1 at (70,  10, 6):  dx 64, dy +10 (left), dz -4  ->  tx -8.9, ty -3.6, range 64.8
 *   tag 2 at (70, -10, 6):  dx 64, dy -10 (right), dz -4 ->  tx +8.9, ty -3.6, range 64.8
 * </pre>
 */
class LimelightBridgeTest {

    private static AprilTagLayout twoTags() {
        return AprilTagLayout.builder()
                .addTag(1, 70.0, 10.0, 6.0, Math.PI, 5.0)
                .addTag(2, 70.0, -10.0, 6.0, Math.PI, 5.0)
                .build();
    }

    /** A started, polled device: one 50 ms tick captures a frame and offers it to a poll. */
    private static FakeLimelight3A polled(AprilTagLayout layout) {
        FakeLimelight3A device = FakeLimelight3A.builder(layout, CameraMount.front(6.0, 10.0, 0.0),
                65.0, 51.0).build();
        device.setRobotPose(0.0, 0.0, 0.0);
        device.start();
        device.update(0.05);
        return device;
    }

    private static AprilTagSingleDetection only(List<AprilTagDetection> detections) {
        assertEquals(1, detections.size(), "exactly one detection was asked for");
        return (AprilTagSingleDetection) detections.get(0);
    }

    @Test
    void aFiducialBecomesAnSdkDetectionWithTheGeometryIntact() {
        AprilTagConfig config = new AprilTagConfig();
        config.applyJson("{\"tagNames\":\"{\\\"1\\\":\\\"R_HIVE\\\"}\"}");

        LLResult result = polled(twoTags()).getLatestResult();
        List<AprilTagDetection> detections = LimelightBridge.toDetections(result, config);

        assertEquals(2, detections.size(), "both tags in view become one detection each");

        AprilTagSingleDetection left = (AprilTagSingleDetection) detections.get(0);
        assertEquals(1, left.id, "the Fiducial array is in ascending id order");
        assertEquals("R_HIVE", left.metadata.name, "tagNames renames a Limelight fiducial");
        assertEquals(5.0, left.metadata.tagsize, 1e-9);
        assertEquals(DistanceUnit.INCH, left.metadata.distanceUnit);

        // Metres in the payload, inches everywhere downstream.
        assertEquals(64.815, left.ftcPose.range, 0.3, "range is the camera-to-tag distance");
        assertEquals(-8.88, left.ftcPose.bearing, 0.75, "tx is negative left of the image centre");
        assertEquals(-3.58, left.ftcPose.elevation, 0.75, "ty is negative below the centre");
        assertEquals(-3.58, left.ftcPose.pitch, 0.75, "the pose mirrors the elevation too");
        assertEquals(-8.88, left.ftcPose.yaw, 0.75, "and the bearing into yaw");

        assertEquals(0, left.hamming, "the Limelight does not report hamming");
        assertEquals(100f, left.decisionMargin, 1e-6, "no margin gate: 100 clears any default");
        assertEquals(0.0, left.ftcPose.roll, 1e-9, "roll comes from skew, which is 0 here");
        assertNotNull(left.corners);
        assertEquals(4, left.corners.length, "the four pts corners survive the reshape");
        assertTrue(left.frameAcquisitionNanoTime > 0, "the result carries its capture stamp");

        AprilTagSingleDetection right = (AprilTagSingleDetection) detections.get(1);
        assertEquals(2, right.id);
        assertEquals(8.88, right.ftcPose.bearing, 0.75, "the other tag is right of the centre");
        assertEquals(left.ftcPose.range, right.ftcPose.range, 0.3, "symmetric about the axis");
    }

    @Test
    void theBridgeSatisfiesTheTargetingRulesEndToEnd() {
        AprilTagConfig config = new AprilTagConfig();
        config.applyJson("{\"alliance\":\"RED\","
                + "\"tagNames\":\"{\\\"1\\\":\\\"R_HIVE\\\",\\\"2\\\":\\\"B_CELL\\\"}\"}");

        List<AprilTagDetection> detections =
                LimelightBridge.toDetections(polled(twoTags()).getLatestResult(), config);

        AprilTagDetection best = AprilTagTargeting.bestTarget(detections, config);
        assertNotNull(best, "the red half of the view is scorable on RED");
        assertEquals(1, ((AprilTagSingleDetection) best).id, "the blue-tagged fiducial is gated out");

        config.applyJson("{\"alliance\":\"BLUE\"}");
        AprilTagDetection blueBest = AprilTagTargeting.bestTarget(detections, config);
        assertEquals(2, ((AprilTagSingleDetection) blueBest).id,
                "the alliance rule reads the name the bridge put on the detection");

        config.applyJson("{\"minDecisionMargin\":150}");
        assertNull(AprilTagTargeting.bestTarget(detections, config),
                "a margin Limelight never reports cannot be demanded of it");
    }

    @Test
    void aFrameWithNoTagsOrNoPollYieldsNothingRatherThanGarbage() {
        FakeLimelight3A device = FakeLimelight3A.builder(AprilTagLayout.empty(),
                CameraMount.front(6.0, 10.0, 0.0), 65.0, 51.0).build();
        device.setRobotPose(0.0, 0.0, 0.0);
        device.start();
        device.update(0.05);

        AprilTagConfig config = new AprilTagConfig();
        List<AprilTagDetection> none =
                LimelightBridge.toDetections(device.getLatestResult(), config);
        assertTrue(none.isEmpty(), "an empty field is an empty detection list");

        FakeLimelight3A untouched = FakeLimelight3A.builder(AprilTagLayout.empty(),
                CameraMount.front(6.0, 10.0, 0.0), 65.0, 51.0).build();
        assertTrue(LimelightBridge.toDetections(untouched.getLatestResult(), config).isEmpty(),
                "before the first poll the result is the SDK's empty one, never null");
    }

    @Test
    void nullsAreDefensive() {
        AprilTagConfig config = new AprilTagConfig();
        LLResult result = polled(twoTags()).getLatestResult();

        assertTrue(LimelightBridge.toDetections(null, config).isEmpty());
        assertTrue(LimelightBridge.toDetections(result, null).isEmpty());
        assertNull(LimelightBridge.toDetection(null, config, 0L));
        assertNull(LimelightBridge.toDetection(
                result.getFiducialResults().get(0), null, 0L));
        assertFalse(LimelightBridge.toDetections(result, config).isEmpty(),
                "the result itself is still good");
    }

    /* ---------------- patchTagSize: the "push tag size" button's whole job ---------------- */

    @Test
    void thePatchRewritesTheSizeAndCarriesEveryOtherSettingThrough() throws Exception {
        // The device stores millimetres - the unit the Limelight's own slider is documented in.
        String pipeline = "{\"tagSize\":127.0,\"fam\":\"apriltag3_36h11_classic\","
                + "\"exposure\":5,\"type\":\"pipe_fiducial\"}";

        LimelightBridge.PipelinePatch patch = LimelightBridge.patchTagSize(pipeline, 6.0);

        assertNotNull(patch.json, patch.message);
        JSONObject out = new JSONObject(patch.json);
        assertEquals(152.4, out.getDouble("tagSize"), 1e-9, "6.0 in is 152.4 mm");
        assertEquals("apriltag3_36h11_classic", out.getString("fam"), "the family rides along");
        assertEquals(5, out.getInt("exposure"), "so does the exposure");
        assertEquals("pipe_fiducial", out.getString("type"), "and the pipeline type");
        assertTrue(patch.message.contains("tagSize") && patch.message.contains("mm"),
                "the message says which key and which unit: " + patch.message);
        assertFalse(patch.message.contains("uploaded"), "the bridge does not upload, only patches");
    }

    @Test
    void theUnitOfTheValueAlreadyThereDecidesHowItIsWrittenBack() throws Exception {
        // Inches: 4.0 cannot be 4.0 mm of a real tag, so the inches reading wins over mm.
        LimelightBridge.PipelinePatch inches = LimelightBridge.patchTagSize("{\"tagSize\":4.0}", 6.0);
        assertNotNull(inches.json, inches.message);
        assertEquals(6.0, new JSONObject(inches.json).getDouble("tagSize"), 1e-9,
                "an inch pipeline is rewritten in inches, not rescaled to mm");

        // Millimetres: 127 is nowhere near 6.0 in any other unit, but is a clean 5 in of mm.
        LimelightBridge.PipelinePatch mm = LimelightBridge.patchTagSize("{\"tagSize\":127.0}", 6.0);
        assertEquals(152.4, new JSONObject(mm.json).getDouble("tagSize"), 1e-9);

        // Metres: 0.127 reads as 5.0 in only when taken as metres.
        LimelightBridge.PipelinePatch metres = LimelightBridge.patchTagSize("{\"tagSize\":0.127}", 4.0);
        assertNotNull(metres.json, metres.message);
        assertEquals(0.1016, new JSONObject(metres.json).getDouble("tagSize"), 1e-9,
                "4.0 in is 0.1016 m");
    }

    @Test
    void aKeyThatSaysTagBeatsOneThatMerelySaysSize() throws Exception {
        String pipeline = "{\"size\":5.0,\"tagsize\":127.0}";

        LimelightBridge.PipelinePatch patch = LimelightBridge.patchTagSize(pipeline, 6.0);

        assertNotNull(patch.json, patch.message);
        JSONObject out = new JSONObject(patch.json);
        assertEquals(152.4, out.getDouble("tagsize"), 1e-9, "the tag key was the one written");
        assertEquals(5.0, out.getDouble("size"), 1e-9, "the other size key was left alone");
    }

    @Test
    void aNestedSizeKeyIsPatchedWhereItSits() throws Exception {
        String pipeline = "{\"type\":\"pipe_fiducial\",\"adv\":"
                + "{\"tagsize\":127.0,\"fam\":\"36h11\"}}";

        LimelightBridge.PipelinePatch patch = LimelightBridge.patchTagSize(pipeline, 5.0);

        assertNotNull(patch.json, patch.message);
        JSONObject out = new JSONObject(patch.json);
        JSONObject adv = out.getJSONObject("adv");
        assertEquals(127.0, adv.getDouble("tagsize"), 1e-9, "5.0 in in millimetres, unchanged");
        assertEquals("36h11", adv.getString("fam"), "its sibling survives");
        assertEquals("pipe_fiducial", out.getString("type"), "so does the top level");
    }

    @Test
    void anythingItCannotIdentifyIsRefusedWithWhatItFoundInstead() {
        LimelightBridge.PipelinePatch noKey = LimelightBridge.patchTagSize(
                "{\"color\":\"blue\",\"exposure\":5}", 5.0);
        assertNull(noKey.json, "nothing may be written without a key to write to");
        assertTrue(noKey.message.contains("no size-like key"), noKey.message);
        assertTrue(noKey.message.contains("color"), "the refusal lists what the file has: "
                + noKey.message);

        LimelightBridge.PipelinePatch nonsense = LimelightBridge.patchTagSize(
                "{\"tagSize\":0.005}", 5.0);
        assertNull(nonsense.json, "0.005 is not a tag length in any unit");
        assertTrue(nonsense.message.contains("nothing was changed"), nonsense.message);

        assertNull(LimelightBridge.patchTagSize(null, 5.0).json);
        assertNull(LimelightBridge.patchTagSize("", 5.0).json);
        assertNull(LimelightBridge.patchTagSize("{not json", 5.0).json);
    }
}
