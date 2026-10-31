package org.firstinspires.ftc.teamcode;

import com.qualcomm.hardware.limelightvision.LLResult;
import com.qualcomm.hardware.limelightvision.LLResultTypes.FiducialResult;

import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import org.firstinspires.ftc.robotcore.external.navigation.Pose3D;
import org.firstinspires.ftc.robotcore.external.navigation.Position;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagMetadata;
import org.firstinspires.ftc.vision.apriltag.AprilTagPoseFtc;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Turns a Limelight 3A result into the SDK detection type everything downstream already reads.
 *
 * <p>The dashboard, {@link AprilTagTargeting} and the state JSON all consume
 * {@code List<AprilTagDetection>}, so a Limelight and a USB webcam are interchangeable from the
 * targeting rules down. Nothing above this class knows which one it is looking at.
 *
 * <p>What the Limelight does not report, and how it is filled in:
 * <ul>
 *   <li><b>hamming</b> - not exposed over the SDK API, so 0 (assume clean).</li>
 *   <li><b>decisionMargin</b> - likewise; 100 means "no margin gate", so a team's
 *       {@code minDecisionMargin} never silently drops every Limelight detection.</li>
 *   <li><b>roll</b> - taken from the Limelight's own {@code skew}, which is the tag's
 *       rotation in the image: the same quantity the |roll| rule gates on.</li>
 *   <li><b>cluster detections</b> - a Limelight reports fiducials one by one, so there is no
 *       cluster object; {@code minClusterPercent} simply has nothing to act on in this mode.</li>
 * </ul>
 */
public final class LimelightBridge {

    /** Limelight translation is in metres (LLResult.createPose3DRobot). */
    private static final double METERS_TO_INCHES = 39.37007874015748;

    /** The SDK's own margin scale: high enough that the gate passes, low enough to format. */
    private static final float MARGIN_NOT_REPORTED = 100f;

    private LimelightBridge() {
    }

    /** {@return every fiducial in this result as SDK detections, or an empty list}. */
    public static List<AprilTagDetection> toDetections(LLResult result, AprilTagConfig config) {
        if (result == null || config == null) {
            return Collections.emptyList();
        }
        List<FiducialResult> fiducials = result.getFiducialResults();
        if (fiducials == null || fiducials.isEmpty()) {
            return Collections.emptyList();
        }
        long acq = result.getControlHubTimeStampNanos();
        List<AprilTagDetection> out = new ArrayList<>(fiducials.size());
        for (FiducialResult f : fiducials) {
            AprilTagDetection d = toDetection(f, config, acq == 0 ? System.nanoTime() : acq);
            if (d != null) {
                out.add(d);
            }
        }
        return out;
    }

    /** {@return one fiducial as an SDK single detection, or null when the entry is unusable}. */
    public static AprilTagSingleDetection toDetection(FiducialResult f, AprilTagConfig config,
                                                       long acqNanos) {
        if (f == null || config == null) {
            return null;
        }
        int id = f.getFiducialId();
        String name = config.tagName(id);

        double[] pos = cameraSpaceInches(f);
        double bearing = f.getTargetXDegrees();
        double elevation = f.getTargetYDegrees();
        double range = Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
        if (range == 0.0) {
            // No t6t_cs in the payload: fall back to the angle pair so range-gating still has
            // something monotonic rather than a bogus "at the camera" reading.
            range = Double.NaN;
        }

        AprilTagPoseFtc pose = new AprilTagPoseFtc(
                pos[0], pos[1], pos[2],
                bearing, elevation, f.getSkew(),
                range, bearing, elevation);

        Point centre = new Point(f.getTargetXPixels(), f.getTargetYPixels());
        AprilTagMetadata metadata = new AprilTagMetadata(
                id, name != null ? name : ("ID " + id), config.tagSizeIn, DistanceUnit.INCH);

        return new AprilTagSingleDetection(id, 0, MARGIN_NOT_REPORTED,
                centre, corners(f, centre), metadata, pose, null, null,
                acqNanos, DistanceUnit.INCH);
    }

    /** {@return the target's position in the camera frame, in inches}. */
    static double[] cameraSpaceInches(FiducialResult f) {
        Pose3D target = f.getTargetPoseCameraSpace();
        if (target == null || target.getPosition() == null) {
            return new double[] {0.0, 0.0, 0.0};
        }
        Position p = target.getPosition();
        if (p.unit == null) {
            return new double[] {p.x * METERS_TO_INCHES, p.y * METERS_TO_INCHES, p.z * METERS_TO_INCHES};
        }
        double scale = p.unit.toInches(1.0);
        return new double[] {p.x * scale, p.y * scale, p.z * scale};
    }

    /** {@return the four pixel corners, or a one-pixel quad when the payload has no pts}. */
    static Point[] corners(FiducialResult f, Point centre) {
        List<List<Double>> pts = f.getTargetCorners();
        if (pts != null && pts.size() >= 4) {
            Point[] out = new Point[4];
            for (int i = 0; i < 4; i++) {
                List<Double> c = pts.get(i);
                if (c == null || c.size() < 2) {
                    return fallbackCorners(centre);
                }
                out[i] = new Point(c.get(0), c.get(1));
            }
            return out;
        }
        return fallbackCorners(centre);
    }

    private static Point[] fallbackCorners(Point centre) {
        double x = centre.x;
        double y = centre.y;
        return new Point[] {
                new Point(x - 1, y - 1), new Point(x + 1, y - 1),
                new Point(x + 1, y + 1), new Point(x - 1, y + 1)
        };
    }

    /* ---------------- the dashboard's "push tag size" button ---------------- */

    /** Outcome of {@link #patchTagSize}: the rewritten pipeline, or null with the reason. */
    public static final class PipelinePatch {
        /** The pipeline JSON with only the tag size changed; null when nothing was changed. */
        public final String json;
        /** What happened, in dashboard words: what was found, what it became, or why not. */
        public final String message;

        private PipelinePatch(String json, String message) {
            this.json = json;
            this.message = message;
        }
    }

    /**
     * The units a Limelight pipeline might store its tag size in, best guess first.
     *
     * <p>The device's own UI documents the AprilTag slider in millimetres ("165.1"), so mm is
     * assumed whenever the value fits it; the rest exist so a pipeline saved in inches, centimetres
     * or metres is recognised as such rather than rewritten in the wrong scale.
     */
    private static final String[] UNITS = {"mm", "in", "cm", "m"};
    private static final double[] INCHES_PER_UNIT = {1.0 / 25.4, 1.0, 1.0 / 2.54, 1.0 / 0.0254};
    private static final int[] DECIMALS = {1, 3, 2, 4};
    /**
     * What still reads as a tag length once a value is taken in any of those units.
     *
     * <p>Half an inch is the floor because it is what separates the two readings that actually
     * occur: a value of {@code 4.0} is a 4 in tag, never a 4 mm one, so the millimetres reading
     * of anything under half an inch is rejected rather than guessed at.
     */
    private static final double MIN_TAG_IN = 0.5;
    private static final double MAX_TAG_IN = 48.0;

    private static final class Candidate {
        final JSONObject owner; // the object the key lives in, so a nested key patches in place
        final String key;
        final double value;

        Candidate(JSONObject owner, String key, double value) {
            this.owner = owner;
            this.key = key;
            this.value = value;
        }
    }

    /**
     * Rewrites one key of a Limelight pipeline JSON: its tag size, set to {@code tagSizeIn}.
     *
     * <p>The button behind this reads the device's own pipeline file, changes this one number and
     * hands the whole thing back, so every other setting on the camera - family, exposure,
     * resolution, calibration - is carried through untouched. Which key holds the tag size is not
     * documented, so it is looked up (a numeric key whose name mentions "size") and the unit is
     * inferred from the value already there, preferring an exact match against the dashboard's own
     * tag size and falling back to the millimetres the Limelight UI shows. When neither is
     * convincing nothing is written: the message reports what was found instead.
     *
     * @param pipelineJson the device's pipeline file, as {@code GET /pipeline-atindex} returns it
     * @param tagSizeIn    the tag side length this dashboard has been ranging with
     * @return the patched JSON with its message, or json == null with the refusal
     */
    public static PipelinePatch patchTagSize(String pipelineJson, double tagSizeIn) {
        if (pipelineJson == null || pipelineJson.trim().isEmpty()) {
            return reject("the Limelight returned an empty pipeline");
        }
        JSONObject root;
        try {
            root = new JSONObject(pipelineJson);
        } catch (JSONException e) {
            return reject("the Limelight's pipeline is not JSON: " + e);
        }

        List<Candidate> found = new ArrayList<>();
        collectSizeKeys(root, found);
        java.util.Iterator<String> scan = root.keys();
        while (scan.hasNext()) { // one level of nesting: the UI groups settings under tabs
            Object v = root.opt(scan.next());
            if (v instanceof JSONObject) {
                collectSizeKeys((JSONObject) v, found);
            }
        }
        if (found.isEmpty()) {
            return reject("no size-like key in the pipeline (top level has: " + keyNames(root) + ")");
        }

        // A key that says "tag" beats one that merely says "size", without reordering the rest.
        List<Candidate> cands = new ArrayList<>(found.size());
        for (Candidate c : found) {
            if (c.key.toLowerCase(Locale.US).contains("tag")) {
                cands.add(c);
            }
        }
        for (Candidate c : found) {
            if (!c.key.toLowerCase(Locale.US).contains("tag")) {
                cands.add(c);
            }
        }

        Candidate chosen = null;
        int unit = -1;
        // Tier 1: the value already equals the dashboard's tag size measured in one of the units -
        // that fixes both the key and the unit beyond argument.
        search:
        for (Candidate c : cands) {
            for (int u = 0; u < UNITS.length; u++) {
                double want = tagSizeIn / INCHES_PER_UNIT[u];
                if (want > 0 && Math.abs(c.value - want) <= 0.01 * want) {
                    chosen = c;
                    unit = u;
                    break search;
                }
            }
        }
        if (chosen == null) {
            // Tier 2: the device's own value, read in each unit, has to be a plausible tag length.
            chosen = cands.get(0);
            for (int u = 0; u < UNITS.length; u++) {
                double inches = chosen.value * INCHES_PER_UNIT[u];
                if (inches >= MIN_TAG_IN && inches <= MAX_TAG_IN) {
                    unit = u;
                    break;
                }
            }
            if (unit < 0) {
                return reject("key '" + chosen.key + "' = " + chosen.value
                        + " is not a tag size in mm, in, cm or m - nothing was changed");
            }
        }

        double replacement = tagSizeIn / INCHES_PER_UNIT[unit];
        replacement = Math.round(replacement * Math.pow(10, DECIMALS[unit]))
                / Math.pow(10, DECIMALS[unit]);
        try {
            chosen.owner.put(chosen.key, replacement);
        } catch (JSONException e) {
            return reject("could not rewrite '" + chosen.key + "': " + e);
        }
        String message = String.format(Locale.US, "%s %s -> %s %s (%.2f in)",
                chosen.key, fixed(chosen.value, unit), fixed(replacement, unit),
                UNITS[unit], tagSizeIn);
        String encoded = root.toString();
        if (encoded == null) {
            return reject("the patched pipeline would not encode");
        }
        return new PipelinePatch(encoded, message);
    }

    /** Numeric keys whose name mentions a size, at one level of the pipeline tree. */
    private static void collectSizeKeys(JSONObject owner, List<Candidate> out) {
        java.util.Iterator<String> it = owner.keys();
        while (it.hasNext()) {
            String key = it.next();
            if (!key.toLowerCase(Locale.US).contains("size")) {
                continue;
            }
            Object v = owner.opt(key);
            if (!(v instanceof Number)) {
                continue;
            }
            double d = ((Number) v).doubleValue();
            if (d >= 0.001 && d <= 500.0) {
                out.add(new Candidate(owner, key, d));
            }
        }
    }

    private static String keyNames(JSONObject root) {
        StringBuilder sb = new StringBuilder();
        java.util.Iterator<String> it = root.keys();
        int shown = 0;
        while (it.hasNext() && shown < 8) {
            if (shown > 0) {
                sb.append(", ");
            }
            sb.append(it.next());
            shown++;
        }
        return sb.length() == 0 ? "nothing" : sb.toString();
    }

    private static String fixed(double value, int unit) {
        return String.format(Locale.US, "%." + DECIMALS[unit] + "f", value);
    }

    private static PipelinePatch reject(String why) {
        return new PipelinePatch(null, why);
    }
}
