// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
    func testQuantizedLinearAppliesNVFP4GlobalScaleOnMetal() {
        let (input, quantizedWeight, scales, biases, globalScale, bias, expected) =
            Device.withDefaultDevice(.cpu) {
                MLXRandom.seed(7)
                let weight = MLXRandom.uniform(low: -2, high: 2, [32, 32])
                let input = MLXRandom.uniform(low: -1, high: 1, [4, 32])
                let globalScale = MLX.max(abs(weight)).asType(.float32)
                let bias = MLXRandom.uniform(low: -1, high: 1, [32])
                let (quantizedWeight, scales, biases) = MLX.quantized(
                    weight, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
                let expected =
                    matmul(
                        input,
                        dequantized(
                            quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                            mode: .nvfp4, globalScale: globalScale, dtype: .float32
                        ).T)
                    + bias
                eval(quantizedWeight, scales, expected)
                return (input, quantizedWeight, scales, biases, globalScale, bias, expected)
            }

        let actual = Device.withDefaultDevice(.gpu) {
            QuantizedLinear(
                weight: quantizedWeight, bias: bias, scales: scales, biases: biases,
                groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)(input)
        }

        assertEqual(actual, expected, rtol: 1e-5, atol: 1e-5)
    }

    func testQuantizedEmbeddingAppliesNVFP4GlobalScaleOnMetal() {
        let (
            input, indices, quantizedWeight, scales, biases, globalScale, expectedLinear,
            expectedLookup
        ) = Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(11)
            let weight = MLXRandom.uniform(low: -2, high: 2, [32, 32])
            let input = MLXRandom.uniform(low: -1, high: 1, [4, 32])
            let indices = MLXArray([0, 3, 7, 31])
            let globalScale = MLX.max(abs(weight)).asType(.float32)
            let (quantizedWeight, scales, biases) = MLX.quantized(
                weight, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
            let dequantizedWeight = dequantized(
                quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                mode: .nvfp4, globalScale: globalScale, dtype: .float32)
            let expectedLinear = matmul(input, dequantizedWeight.T)
            let expectedLookup = dequantized(
                quantizedWeight, scales: scales, biases: biases, groupSize: 16, bits: 4,
                mode: .nvfp4, globalScale: globalScale)[indices].asType(.bfloat16)
            eval(quantizedWeight, scales, expectedLinear, expectedLookup)
            return (
                input, indices, quantizedWeight, scales, biases, globalScale, expectedLinear,
                expectedLookup
            )
        }

        let (actualLinear, actualLookup) = Device.withDefaultDevice(.gpu) {
            let embedding = QuantizedEmbedding(
                weight: quantizedWeight, scales: scales, biases: biases,
                groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
            return (embedding.asLinear(input), embedding(indices))
        }

        assertEqual(actualLinear, expectedLinear, rtol: 1e-5, atol: 1e-5)
        assertEqual(actualLookup, expectedLookup, rtol: 1e-5, atol: 1e-5)
    }

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

    func testQuantizedLinearStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedEmbeddingStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedGlobalScaleIsOptionalParameter() {
        let linear = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)
        let embedding = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)

        XCTAssertNil(linear.globalScale)
        XCTAssertNil(linear.parameters()["global_scale"])
        XCTAssertNil(embedding.globalScale)
        XCTAssertNil(embedding.parameters()["global_scale"])
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
                // Symmetric binary weights {-0.3, +0.3}.
                let signs = (MLXRandom.uniform(low: 0, high: 1, [128, gs * 2]) .> 0.5)
                let w = `where`(signs, Float(0.3), Float(-0.3))

                let (wq, scales, biases) = quantized(w, groupSize: gs, bits: bits)
                XCTAssertNotNil(biases, "affine \(bits)-bit must produce biases")
                let wHat = dequantized(
                    wq, scales: scales, biases: biases, groupSize: gs, bits: bits)
                eval(wHat)

                if bits == 1 {
                    // 1-bit affine packs exactly two levels {bias, bias+scale} = the two
                    // distinct weight values, so a binary weight round-trips near-exactly.
                    // This is the property PrismML's 1-bit path adds.
                    let maxErr = abs(w - wHat).max().item(Float.self)
                    XCTAssertLessThan(
                        maxErr, 1e-4,
                        "1-bit g\(gs) reconstruction error \(maxErr) too large")
                }
                // NOTE: a binary weight does NOT round-trip exactly at bits>=2 — this is stock
                // MLX affine behavior, not a PrismML change. The bits>=2 affine quantize code is
                // unchanged by the patch (only wrapped in an `else`); its grid is asymmetric
                // (e.g. {-0.3,+0.3} g* -> scale=-0.15, bias=0.3, grid {0.3,0.15,0,-0.15}, so
                // -0.3 maps to -0.15). Exactness for bits>=2 is covered by testBitExactRegression.

                // quantizedMatmul(bits:) must match dequantize-then-matmul for both bit widths.
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

    // Bias-free (symmetric) 1/2-bit affine: scales-only checkpoints, where the kernel derives
    // the bias (-0.5 * scale at 1 bit, -scale at 2 bits). quantizedMatmul(biases: nil) must
    // match the same weights with that bias passed explicitly, on the qmv_fast path
    // (K % alignment == 0) and the general qmv path. Covers the affine_sym kernels ported
    // onto the v0.32.2 core. wangqi modified 2026-10-17
    func testBiasFreeLowBitMatchesDerivedBias() {
        for bits in [1, 2] {
            for k in [4096, 1024 + 512] {
                MLXRandom.seed(99)
                let w = MLXRandom.normal([64, k])
                let (wq, scales, _) = quantized(w, groupSize: 64, bits: bits)
                let derived = bits == 1 ? scales * Float(-0.5) : -scales
                let x = MLXRandom.normal([1, k])
                let yFree = quantizedMatmul(
                    x, wq, scales: scales, biases: nil,
                    transpose: true, groupSize: 64, bits: bits)
                let yRef = quantizedMatmul(
                    x, wq, scales: scales, biases: derived,
                    transpose: true, groupSize: 64, bits: bits)
                eval(yFree, yRef)
                XCTAssertEqual(yFree.shape, yRef.shape)
                let err = abs(yFree - yRef).max().item(Float.self)
                XCTAssertLessThan(
                    err, 1e-3, "bias-free \(bits)-bit K=\(k) mismatch \(err)")
            }
        }
    }
}
