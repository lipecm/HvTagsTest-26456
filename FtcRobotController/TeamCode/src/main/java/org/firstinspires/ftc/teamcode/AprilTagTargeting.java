package org.firstinspires.ftc.teamcode;

import org.firstinspires.ftc.vision.apriltag.AprilTagClusterDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagSingleDetection;

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
    public static AprilTagDetection bestTarget(java.util.List<AprilTagDetection> detections, AprilTagConfig config) {
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
}
