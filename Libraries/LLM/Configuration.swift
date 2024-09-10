// Copyright © 2024 Apple Inc.

import Foundation

public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case float(Float)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .float(let v): try container.encode(v)
        }
    }
}

public actor ModelTypeRegistry {
    @Sendable private static func createLlamaModel(url: URL) throws -> LLMModel {
        let configuration = try JSONDecoder().decode(
            LlamaConfiguration.self, from: Data(contentsOf: url))
        return LlamaModel(configuration)
    }

    private var creators: [String: @Sendable (URL) throws -> any LLMModel] = [
        "mistral": createLlamaModel,
        "llama": createLlamaModel,
        "phi": { url in
            let configuration = try JSONDecoder().decode(
                PhiConfiguration.self, from: Data(contentsOf: url))
            return PhiModel(configuration)
        },
        "phi3": { url in
            let configuration = try JSONDecoder().decode(
                Phi3Configuration.self, from: Data(contentsOf: url))
            return Phi3Model(configuration)
        },
        "gemma": { url in
            let configuration = try JSONDecoder().decode(
                GemmaConfiguration.self, from: Data(contentsOf: url))
            return GemmaModel(configuration)
        },
        "gemma2": { url in
            let configuration = try JSONDecoder().decode(
                Gemma2Configuration.self, from: Data(contentsOf: url))
            return Gemma2Model(configuration)
        },
        "qwen2": { url in
            let configuration = try JSONDecoder().decode(
                Qwen2Configuration.self, from: Data(contentsOf: url))
            return Qwen2Model(configuration)
        },
        "starcoder2": { url in
            let configuration = try JSONDecoder().decode(
                Starcoder2Configuration.self, from: Data(contentsOf: url))
            return Starcoder2Model(configuration)
        },
        "cohere": { url in
            let configuration = try JSONDecoder().decode(
                CohereConfiguration.self, from: Data(contentsOf: url))
            return CohereModel(configuration)
        },
        "openelm": { url in
            let configuration = try JSONDecoder().decode(
                OpenElmConfiguration.self, from: Data(contentsOf: url))
            return OpenELMModel(configuration)
        },
        "internlm2": { url in
            let configuration = try JSONDecoder().decode(
                InternLM2Configuration.self, from: Data(contentsOf: url))
            return InternLM2Model(configuration)
        },
    ]

    func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any LLMModel
    ) {
        creators[type] = creator
    }

    public func getCreator(for type: String) -> (@Sendable (URL) throws -> LLMModel)? {
        creators[type]
    }
}

public struct ModelType: RawRepresentable, Codable, Sendable {
    public let rawValue: String
    private static let registry = ModelTypeRegistry()

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    private static func createLlamaModel(url: URL) throws -> LLMModel {
        let configuration = try JSONDecoder().decode(
            LlamaConfiguration.self, from: Data(contentsOf: url))
        return LlamaModel(configuration)
    }

    public static func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any LLMModel
    ) async {
        await registry.registerModelType(type, creator: creator)
    }

    public func createModel(configuration: URL) async throws -> LLMModel {
        guard let creator = await ModelType.registry.getCreator(for: rawValue) else {
            throw LLMError(message: "Unsupported model type.")
        }
        return try creator(configuration)
    }

    //    public func createModelSynchronous(configuration: URL) throws -> any LLMModel {
    //        // 使用 DispatchQueue 来同步执行异步操作
    //        let semaphore = DispatchSemaphore(value: 0)
    //        var result: Result<any LLMModel, Error>?
    //
    //        DispatchQueue.global().async {
    //            Task {
    //                do {
    //                    let model = try await self.createModel(configuration: configuration)
    //                    result = .success(model)
    //                } catch {
    //                    result = .failure(error)
    //                }
    //                semaphore.signal()
    //            }
    //        }
    //
    //        semaphore.wait()
    //
    //        guard let finalResult = result else {
    //            fatalError("Unexpected state: result not set")
    //        }
    //
    //        return try finalResult.get()
    //    }

    public func createModelSynchronous(configuration: URL) throws -> any LLMModel {
        let semaphore = DispatchSemaphore(value: 0)
        var resultModel: (any LLMModel)?
        var resultError: Error?

        Task {
            do {
                let model = try await self.createModel(configuration: configuration)
                resultModel = model
            } catch {
                resultError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = resultError {
            throw error
        }
        guard let model = resultModel else {
            throw LLMError(message: "Failed to create model")
        }
        return model
    }

}
public struct BaseConfiguration: Codable, Sendable {
    public let modelType: ModelType

    public struct Quantization: Codable, Sendable {
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        let groupSize: Int
        let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
        }
    }

    public var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}
