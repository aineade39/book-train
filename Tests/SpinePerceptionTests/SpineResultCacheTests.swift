import Foundation
import XCTest

@testable import SpinePerception

final class SpineResultCacheTests: XCTestCase {
    func testValueForIdIsNilBeforeAnyStore() async {
        let cache = SpineResultCache<String>()
        let id = UUID()
        let value = await cache.value(for: id)
        XCTAssertNil(value)
    }

    func testStoreThenValueForReturnsTheStoredValue() async {
        let cache = SpineResultCache<String>()
        let id = UUID()
        await cache.store("assembled text", for: id)
        let value = await cache.value(for: id)
        XCTAssertEqual(value, "assembled text")
    }

    func testDistinctIdsDoNotCollide() async {
        let cache = SpineResultCache<Int>()
        let a = UUID()
        let b = UUID()
        await cache.store(1, for: a)
        await cache.store(2, for: b)
        let va = await cache.value(for: a)
        let vb = await cache.value(for: b)
        XCTAssertEqual(va, 1)
        XCTAssertEqual(vb, 2)
    }

    func testComputeIfMissingComputesOnceThenReusesCachedValue() async {
        let cache = SpineResultCache<Int>()
        let id = UUID()
        var computeCount = 0

        let first = await cache.value(for: id) {
            computeCount += 1
            return 42
        }
        let second = await cache.value(for: id) {
            computeCount += 1
            return 99 // should never run -- id is already cached
        }

        XCTAssertEqual(first, 42)
        XCTAssertEqual(second, 42)
        XCTAssertEqual(computeCount, 1)
    }

    func testConcurrentComputeIfMissingForTheSameIdComputesExactlyOnce() async {
        let cache = SpineResultCache<Int>()
        let id = UUID()
        let computeCounter = ActorCounter()

        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    await cache.value(for: id) {
                        Task { await computeCounter.increment() }
                        return 7
                    }
                }
            }
            for await value in group {
                XCTAssertEqual(value, 7)
            }
        }

        // The actor serializes calls to `value(for:computeIfMissing:)`, so
        // only the first of the 20 concurrent callers should ever find the
        // cache empty and invoke `compute`.
        try? await Task.sleep(nanoseconds: 20_000_000)
        let count = await computeCounter.value
        XCTAssertEqual(count, 1)
    }

    func testRemoveAllClearsTheCache() async {
        let cache = SpineResultCache<Int>()
        let id = UUID()
        await cache.store(5, for: id)
        await cache.removeAll()
        let value = await cache.value(for: id)
        XCTAssertNil(value)
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }
}

private actor ActorCounter {
    private(set) var value = 0
    func increment() { value += 1 }
}
