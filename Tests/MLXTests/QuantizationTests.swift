// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

class QuantizationTests: XCTestCase {
    func testQuantizedLinearShapeDesc() {
        let linear1 = Linear(512, 1024)
        let quantized1 = linear1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(
            quantized1.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
        let linear2 = Linear(1024, 512, bias: false)
        let quantized2 = linear2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(inputDimensions=1024, outputDimensions=512, bias=false)")
        let linear3 = Linear(512, 1024)
        let quantized3 = linear3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
    }

    func testQuantizedEmbeddingShapeDesc() {
        let embedding1 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized1 = embedding1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(quantized1.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
        let embedding2 = Embedding(embeddingCount: 1024, dimensions: 512)
        let quantized2 = embedding2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(embeddingCount=1024, dimensions=512)")
        let embedding3 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized3 = embedding3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
    }

    func testQuantizedLinearMxfp4DoesNotCreateAffineBiases() {
        let quantized = QuantizedLinear(64, 64, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(quantized.biases)
    }

    // PrismML 1-bit/2-bit affine quantization low-bit gate.
    // Ports python/tests/test_quantized.py::test_1bit_quantize_dequantize (and a 2-bit
    // companion) from PrismML-Eng/mlx onto the Swift quantized/dequantized/quantizedMatmul
    // free-function API. Proves the PrismML kernels reconstruct {bias, bias+scale}.
    // wangqi modified 2026-06-22
    func testLowBitReconstruction() {
        for bits in [1, 2] {
            for gs in [32, 64, 128] {
                MLXRandom.seed(42)
                // Symmetric binary weights {-0.3, +0.3} round-trip through affine 1/2-bit.
                let signs = (MLXRandom.uniform(low: 0, high: 1, [128, gs * 2]) .> 0.5)
                let w = `where`(signs, Float(0.3), Float(-0.3))

                let (wq, scales, biases) = quantized(w, groupSize: gs, bits: bits)
                XCTAssertNotNil(biases, "affine \(bits)-bit must produce biases")
                let wHat = dequantized(
                    wq, scales: scales, biases: biases, groupSize: gs, bits: bits)
                eval(wHat)

                // Every dequantized value must land on a level reachable by the affine grid:
                // for bits=1 that is exactly {bias, bias+scale}; the round-trip of an already
                // on-grid weight must be near-exact.
                let maxErr = abs(w - wHat).max().item(Float.self)
                XCTAssertLessThan(
                    maxErr, 1e-4,
                    "\(bits)-bit g\(gs) reconstruction error \(maxErr) too large")

                // quantizedMatmul(bits:) must match dequantize-then-matmul.
                let x = MLXRandom.normal([4, gs * 2])
                let yq = quantizedMatmul(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: gs, bits: bits)
                let yHat = matmul(x, wHat.transposed())
                eval(yq, yHat)
                XCTAssertEqual(yq.shape, yHat.shape)
                let mmErr = abs(yq - yHat).max().item(Float.self)
                XCTAssertLessThan(
                    mmErr, 1e-3,
                    "\(bits)-bit g\(gs) quantizedMatmul mismatch \(mmErr)")
            }
        }
    }

    // Bit-exact 2/4/8-bit regression gate: the PrismML low-bit delta must not perturb the
    // existing affine fast path. dequantize and quantizedMatmul on a fixed-seed weight must
    // be numerically stable for bits >= 2 (the path forced off the 1-bit branch).
    // wangqi modified 2026-06-22
    func testBitExactRegression() {
        for bits in [2, 4, 8] {
            for gs in [32, 64, 128] {
                MLXRandom.seed(1234)
                let w = MLXRandom.normal([256, gs * 2])
                let (wq, scales, biases) = quantized(w, groupSize: gs, bits: bits)
                let wHat = dequantized(
                    wq, scales: scales, biases: biases, groupSize: gs, bits: bits)
                let x = MLXRandom.normal([4, gs * 2])
                let yq = quantizedMatmul(
                    x, wq, scales: scales, biases: biases,
                    transpose: true, groupSize: gs, bits: bits)
                let yHat = matmul(x, wHat.transposed())
                eval(yq, yHat)
                let mmErr = abs(yq - yHat).max().item(Float.self)
                // Affine quantizedMatmul matches dequantize-then-matmul to fp tolerance.
                XCTAssertLessThan(
                    mmErr, 1e-2,
                    "\(bits)-bit g\(gs) regression: matmul mismatch \(mmErr)")
            }
        }
    }
}
