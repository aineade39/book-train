import XCTest
@testable import SpineCore

final class DefaultBookSpinesDataRootTests: XCTestCase {
    func testHostBookSpinesDataRootFromSimulatorHome() {
        let home = "/Users/joebr/Library/Developer/CoreSimulator/Devices/ABC/data/Containers/Data/Application/XYZ"
        let root = hostBookSpinesDataRootFromSimulatorHome(home) { path in
            path == "/Users/joebr/ml/book-spines"
        }
        XCTAssertEqual(root, "/Users/joebr/ml/book-spines")
    }

    func testHostBookSpinesDataRootFromSimulatorHomeReturnsNilForNonSimulatorPath() {
        XCTAssertNil(hostBookSpinesDataRootFromSimulatorHome("/var/mobile/Containers/Data/Application/XYZ") { _ in true })
    }

    func testHostBookSpinesDataRootFromSimulatorHomeReturnsNilWhenHostRootMissing() {
        let home = "/Users/joebr/Library/Developer/CoreSimulator/Devices/ABC/data"
        XCTAssertNil(hostBookSpinesDataRootFromSimulatorHome(home) { _ in false })
    }

    func testDefaultBookSpinesDataRootHonorsOverride() {
        XCTAssertEqual(defaultBookSpinesDataRoot(override: "/tmp/custom"), "/tmp/custom")
    }
}
