package org.firstinspires.ftc.teamcode;

import org.firstinspires.ftc.vision.apriltag.AprilTagClusterDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;

import java.util.List;
import java.util.Locale;

/**
 * "Should the robot actually aim at this?" rules for BIOBUZZ-style targets, per the FTC
 * AprilTag Clusters tech tip (ftc-docs.firstinspires.org).
 *
 * Two conditions decide if a cluster is scorable from the robot's camera:
 *   1. Alliance colour - the cluster name starts with R (red) or B (blue); it must match
 *      the alliance the robot is on (config.alliance, or ANY to accept both).
 *   2. Roll - a scorable cell's cluster is roughly right-side up. If |ftcPose.roll| is
 *      less than maxRollDeg (default 90) the cluster is upright; outside that range it is
 *      upside down and not scorable.
 *
 * Extra health filters (optional, default off): percent of cluster tags visible, and the
 * single-tag decision margin. Cluster detections always carry metadata (the SDK needs it
 * to know the tag belongs to a cluster); single detections may have null metadata when the
 * tag family/library has no entry for that id.
 */
public class AprilTagTargeting {

    private AprilTagTargeting() {
    }

    public static boolean isScorable(AprilTagDetection detection, AprilTagConfig config) {
        if (detection == null || detection.ftcPose == null) {
            return false;
        }

        // Rule 2 from the tech tip: upright means |roll| <= threshold.
        if (Math.abs(detection.ftcPose.roll) > config.maxRollDeg) {
            return false;
        }

        if (detection instanceof AprilTagClusterDetection) {
            AprilTagClusterDetection cluster = (AprilTagClusterDetection) detection;
            if (cluster.metadata == null) {
                return false;
            }
            if (cluster.percentClusterFound < config.minClusterPercent) {
                return false;
            }
            // Cluster detections always have metadata, so the alliance check always applies.
            return allianceOk(cluster.metadata.name, config.alliance);
        }

        if (detection instanceof AprilTagSingleDetection) {
            AprilTagSingleDetection single = (AprilTagSingleDetection) detection;
            if (single.decisionMargin < config.minDecisionMargin) {
                return false;
            }
            // Unknown ids have null metadata - no name, so no alliance hint to reject.
            if (single.metadata != null && !allianceOk(single.metadata.name, config.alliance)) {
                return false;
            }
            return true;
        }

        return false;
    }

    /**
     * True when the name does not carry an alliance hint, or when its leading R/B matches
     * the configured alliance. ANY accepts everything.
     */
    static boolean allianceOk(String name, String alliance) {
        if (alliance == null || alliance.equals("ANY") || name == null || name.isEmpty()) {
            return true;
        }
        char hint = Character.toUpperCase(name.charAt(0));
        if (hint != 'R' && hint != 'B') {
            return true; // name has no alliance prefix (e.g. "HIVE CELL") - do not filter
        }
        return hint == alliance.charAt(0);
    }

    /** Best scorable target = closest one; null when nothing is currently scorable. */
    public static AprilTagDetection bestTarget(List<AprilTagDetection> detections, AprilTagConfig config) {
        AprilTagDetection best = null;
        double bestRange = Double.MAX_VALUE;
        if (detections == null) {
            return null;
        }
        for (AprilTagDetection d : detections) {
            if (isScorable(d, config) && d.ftcPose.range < bestRange) {
                best = d;
                bestRange = d.ftcPose.range;
            }
        }
        return best;
    }

    /**
     * One short line saying why nothing is scorable - "no tags in view" is not useful when a tag
     * is plainly on screen and the alliance filter is quietly throwing it away.
     *
     * <p>Explains the closest detection, because that is the one a driver is looking at.
     */
    public static String noTargetReason(List<AprilTagDetection> detections, AprilTagConfig config) {
        if (detections == null || detections.isEmpty()) {
            return "no tags in view";
        }
        AprilTagDetection nearest = null;
        for (AprilTagDetection d : detections) {
            if (d != null && d.ftcPose != null
                    && (nearest == null || d.ftcPose.range < nearest.ftcPose.range)) {
                nearest = d;
            }
        }
        if (nearest == null) {
            return "tags seen but none had a pose";
        }
        String name = labelOf(nearest);
        if (Math.abs(nearest.ftcPose.roll) > config.maxRollDeg) {
            return String.format(Locale.US, "%s: roll %.0f deg is past the %.0f deg limit - "
                    + "the cell looks upside down", name, nearest.ftcPose.roll, config.maxRollDeg);
        }
        if (nearest instanceof AprilTagClusterDetection) {
            AprilTagClusterDetection cluster = (AprilTagClusterDetection) nearest;
            if (cluster.metadata == null) {
                return name + ": the SDK has no metadata for this cluster";
            }
            if (cluster.percentClusterFound < config.minClusterPercent) {
                return String.format(Locale.US, "%s: %d%% of the cluster visible, min is %d%%",
                        name, cluster.percentClusterFound, config.minClusterPercent);
            }
            if (!allianceOk(cluster.metadata.name, config.alliance)) {
                return name + ": " + allianceMismatch(cluster.metadata.name, config.alliance);
            }
        } else if (nearest instanceof AprilTagSingleDetection) {
            AprilTagSingleDetection single = (AprilTagSingleDetection) nearest;
            if (single.decisionMargin < config.minDecisionMargin) {
                return String.format(Locale.US, "%s: decision margin %.0f, min is %.0f",
                        name, single.decisionMargin, config.minDecisionMargin);
            }
            if (single.metadata != null && !allianceOk(single.metadata.name, config.alliance)) {
                return name + ": " + allianceMismatch(single.metadata.name, config.alliance);
            }
        }
        return name + ": nothing scorable, and no rule explains it - report this";
    }

    private static String labelOf(AprilTagDetection d) {
        if (d instanceof AprilTagClusterDetection) {
            AprilTagClusterDetection cluster = (AprilTagClusterDetection) d;
            return cluster.metadata == null ? "cluster" : cluster.metadata.name;
        }
        if (d instanceof AprilTagSingleDetection) {
            AprilTagSingleDetection single = (AprilTagSingleDetection) d;
            return "tag " + single.id + (single.metadata == null ? "" : " " + single.metadata.name);
        }
        return "tag";
    }

    private static String allianceMismatch(String name, String alliance) {
        char hint = Character.toUpperCase(name.charAt(0));
        String mine = alliance == null ? "any" : alliance.toLowerCase(Locale.US);
        return mine + " alliance filter, this one is " + (hint == 'R' ? "red" : "blue");
    }
}
