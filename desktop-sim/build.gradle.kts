// Desktop simulator for the AprilTag dashboard: open this folder (desktop-sim) in
// Android Studio and hit Run on DesktopSimMain, or:  ./gradlew run
// It compiles only the two FTC-SDK-free classes from ../TeamCode plus the sim main.
plugins {
    java
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.json:json:20240303")
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("../TeamCode/src/main/java", "src/main/java"))
            include(
                "org/firstinspires/ftc/teamcode/DesktopSimMain.java",
                "org/firstinspires/ftc/teamcode/AprilTagConfig.java",
                "org/firstinspires/ftc/teamcode/AprilTagDashboardServer.java",
            )
        }
    }
}

application {
    mainClass.set("org.firstinspires.ftc.teamcode.DesktopSimMain")
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release.set(11)
}
