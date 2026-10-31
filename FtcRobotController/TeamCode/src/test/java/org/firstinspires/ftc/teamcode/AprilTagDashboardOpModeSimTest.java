package org.firstinspires.ftc.teamcode;

import android.graphics.Canvas;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.apriltag.AprilTagDetection;
import org.firstinspires.ftc.vision.apriltag.AprilTagProcessor;
import org.horizon36596.simloop.fakehardware.FakeHardwareMap;
import org.horizon36596.simloop.opmode.OpModeRunResult;
import org.horizon36596.simloop.opmode.OpModeRunner;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The real {@link AprilTagDashboardOpMode} driven through SimLoop's {@code OpModeRunner}:
 * the SDK's own init / init_loop / loop / stop order, a {@link FakeHardwareMap} with no webcam
 * on it, and the dashboard answering HTTP while the sim ticks.
 *
 * <p>Two seams are overridden here and nowhere else on the robot:
 * {@code createProcessor()} returns a stub because {@code AprilTagProcessor} loads OpenCV's
 * native library (SimLoop does not install OpenCV), and {@code rebuildPortal()} does nothing
 * because {@code VisionPortal} needs Android's camera stack. Everything else - config,
 * targeting, telemetry, the HTTP server and the state JSON - is the shipping code.
 *
 * <p>Tag geometry is covered by {@link AprilTagSimVisionTest}, which runs the simulator's
 * camera rather than a stub.
 */
class AprilTagDashboardOpModeSimTest {

    /** A processor that owns no native code and never reports a detection. */
    private static final class StubProcessor extends AprilTagProcessor {
        @Override
        public void init(int width, int height, CameraCalibration calibration) {
        }

        @Override
        public Object processFrame(Mat frame, long captureTimeNanos) {
            return null;
        }

        @Override
        public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight,
                                float scaleBmpPxToCanvasPx, float scaleCanvasToHttps, Object userContext) {
        }

        @Override
        public void setDecimation(float decimation) {
        }

        @Override
        public void setPoseSolver(PoseSolver poseSolver) {
        }

        @Override
        public int getPerTagAvgPoseSolveTime() {
            return 0;
        }

        @Override
        public ArrayList<AprilTagDetection> getDetections() {
            return new ArrayList<>();
        }

        @Override
        public ArrayList<AprilTagDetection> getFreshDetections() {
            return new ArrayList<>();
        }
    }

    /** The shipping OpMode with only the two Android/native pieces taken out. */
    private static final class SimOpMode extends AprilTagDashboardOpMode {
        @Override
        protected AprilTagProcessor createProcessor() {
            return new StubProcessor();
        }

        @Override
        protected void rebuildPortal() {
            // No VisionPortal on a plain JVM: the dashboard runs against a closed camera.
        }
    }

    private static boolean portFree(int port) {
        try (ServerSocket probe = new ServerSocket()) {
            probe.setReuseAddress(true);
            probe.bind(new InetSocketAddress(port));
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /** GET /api/state from inside a sim tick. Returns the body; status goes in status[0]. */
    private static String fetchState(int[] status) {
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL("http://127.0.0.1:"
                    + AprilTagDashboardServer.PORT + "/api/state").openConnection();
            conn.setConnectTimeout(500);
            conn.setReadTimeout(500);
            status[0] = conn.getResponseCode();
            InputStream in = status[0] >= 400 ? conn.getErrorStream() : conn.getInputStream();
            if (in == null) {
                return "";
            }
            StringBuilder sb = new StringBuilder(4096);
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) != -1) {
                sb.append(new String(buf, 0, n, StandardCharsets.UTF_8));
            }
            return sb.toString();
        } catch (IOException e) {
            status[0] = -1;
            return "";
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }

    @Test
    void opModeRunsAndServesStateWhileTheSimTicks() {
        Assumptions.assumeTrue(portFree(AprilTagDashboardServer.PORT),
                "port " + AprilTagDashboardServer.PORT + " is in use - stop the desktop "
                        + "dashboard (DesktopSimMain) before running these tests");

        // Deliberately empty: no webcam is registered, so the OpMode must survive the lookup
        // failure and report it rather than dying in init().
        FakeHardwareMap hardwareMap = new FakeHardwareMap();
        AprilTagDashboardOpMode opMode = new SimOpMode();

        int[] lastStatus = new int[] {-1};
        int[] answered = new int[] {0};

        OpModeRunResult result = new OpModeRunner(opMode, hardwareMap)
                .initLoopTicks(3)
                .activeTicks(25)
                .eachTick(dt -> {
                    String body = fetchState(lastStatus);
                    if (lastStatus[0] == 200 && body.contains("camState")) {
                        answered[0]++;
                    }
                })
                .run();

        assertTrue(result.ticks() > 0, "the run completed ticks");
        assertTrue(answered[0] > 0,
                "the dashboard must answer over HTTP while the OpMode is running; last status="
                        + lastStatus[0]);

        List<List<String>> sent = result.telemetry().transmissions();
        assertFalse(sent.isEmpty(), "the Driver Station must see telemetry from the OpMode");

        assertTrue(portFree(AprilTagDashboardServer.PORT),
                "stop() must close the dashboard, or the next run cannot bind 8080");
    }

    @Test
    void stateJsonReportsTheCameraItActuallyGot() {
        Assumptions.assumeTrue(portFree(AprilTagDashboardServer.PORT),
                "port " + AprilTagDashboardServer.PORT + " is in use - stop the desktop "
                        + "dashboard (DesktopSimMain) before running these tests");

        FakeHardwareMap hardwareMap = new FakeHardwareMap();
        AprilTagDashboardOpMode opMode = new SimOpMode();

        int[] lastStatus = new int[] {-1};
        String[] body = new String[1];

        new OpModeRunner(opMode, hardwareMap)
                .initLoopTicks(2)
                .activeTicks(5)
                .eachTick(dt -> {
                    String json = fetchState(lastStatus);
                    if (json.contains("camState")) {
                        body[0] = json;
                    }
                })
                .run();

        assertNotNull(body[0], "no state JSON was served; status=" + lastStatus[0]);
        assertTrue(body[0].contains("\"detections\""), body[0]);
        assertTrue(body[0].contains("\"config\""), body[0]);
        assertTrue(body[0].contains("\"hive\""), "the hive tip block is part of the state");
        assertTrue(body[0].contains("\"isTipped\""), body[0]);
        assertTrue(body[0].contains("\"CLOSED\""),
                "with no VisionPortal the camera reads CLOSED, not a lie about streaming");
    }
}
