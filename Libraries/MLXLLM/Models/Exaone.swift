//
//  Exaone.swift
//  mlx-swift-examples
//
//  Created by John Mai on 2025/6/21.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/exaone.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct ExaoneConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var numLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var vocabularySize: Int
    var ropeTheta: Float
    var layerNormEpsilon: Float
    var numKeyValueHeads: Int
    var headDim: Int?
    var maxPositionEmbeddings: Int?
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case layerNormEpsilon = "layer_norm_epsilon"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.numLayers = try container.decode(Int.self, forKey: .numLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.layerNormEpsilon = try container.decode(Float.self, forKey: .layerNormEpsilon)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
    }
}

// MARK: - Attention Module

private class AttentionModule: Module {
    let args: ExaoneConfiguration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPE

    public init(_ args: ExaoneConfiguration) {
        self.args = args
        let dim = args.hiddenSize
        let nHeads = args.numAttentionHeads
        let nKvHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (dim / nHeads)
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: args.attentionBias)
        _outProj.wrappedValue = Linear(nHeads * headDim, dim, bias: args.attentionBias)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
        )
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, L, args.numAttentionHeads, -1).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, args.numKeyValueHeads, -1).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, args.numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
            (k, v) = cache.update(keys: k, values: v)
        } else {
            q = rope(q)
            k = rope(k)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, D)

        return outProj(out)
    }
}

// MARK: - Attention Wrapper

private class Attention: Module {
    let attention: AttentionModule

    public init(_ args: ExaoneConfiguration) {
        self.attention = AttentionModule(args)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        attention(x, mask: mask, cache: cache)
    }
}

// MARK: - MLP

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc_0") var cFc0: Linear
    @ModuleInfo(key: "c_fc_1") var cFc1: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    public init(_ args: ExaoneConfiguration) {
        let dim = args.hiddenSize
        let hiddenDim = args.intermediateSize

        _cFc0.wrappedValue = Linear(dim, hiddenDim, bias: args.mlpBias)
        _cFc1.wrappedValue = Linear(dim, hiddenDim, bias: args.mlpBias)
        _cProj.wrappedValue = Linear(hiddenDim, dim, bias: args.mlpBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(silu(cFc0(x)) * cFc1(x))
    }
}

// MARK: - Transformer Block

private class TransformerBlock: Module {
    @ModuleInfo(key: "ln_1") var ln1: RMSNorm
    let attn: Attention
    @ModuleInfo(key: "ln_2") var ln2: RMSNorm
    let mlp: MLP

    public init(_ args: ExaoneConfiguration) {
        _ln1.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self.attn = Attention(args)
        _ln2.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        self.mlp = MLP(args)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attn.attention(ln1(x), mask: mask, cache: cache)
        let out = h + mlp(ln2(h))
        return out
    }
}

// MARK: - Model

private class ModelInner: Module {
    @ModuleInfo(key: "wte") var wte: Embedding
    fileprivate let h: [TransformerBlock]
    @ModuleInfo(key: "ln_f") var lnF: RMSNorm

    public init(_ args: ExaoneConfiguration) {
        _wte.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.h = (0 ..< args.numLayers).map { _ in TransformerBlock(args) }
        _lnF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wte(inputs)
        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in self.h.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return lnF(h)
    }
}

public class ExaoneModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let transformer: ModelInner
    let configuration: ExaoneConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: ExaoneConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.numLayers).map { _ in args.numKeyValueHeads }
        self.transformer = ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = transformer(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = transformer.wte.asLinear(out)
        }
        return out
    }
}

// MARK: - LoRA

extension ExaoneModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        transformer.h.map { ($0.attn.attention, ["q_proj", "v_proj"]) }
    }
}
