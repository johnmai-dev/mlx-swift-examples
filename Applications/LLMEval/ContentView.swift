// Copyright © 2024 Apple Inc.

import MarkdownUI
import Metal
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import SwiftUI
import Tokenizers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat

    @State var llm = LLMEvaluator()
    @State var llm2 = LLMEvaluator2()

    @State private var prompt = "Who are you?"

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown

    @ViewBuilder
    func LLMEvaluatorView() -> some View {
        VStack {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Toggle(isOn: $llm.includeWeatherTool) {
                        Text("Include \"get current weather\" tool")
                    }
                    .frame(maxWidth: 350, alignment: .leading)
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                }
            }
            // show the model output
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        if selectedDisplayStyle == .plain {
                            Text(llm.output)
                                .textSelection(.enabled)
                        } else {
                            Markdown(llm.output)
                                .textSelection(.enabled)
                        }
                    }
                    .onChange(of: llm.output) { _, _ in
                        sp.scrollTo("bottom")
                    }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
        }
    }

    @ViewBuilder
    func LLMEvaluator2View() -> some View {
        VStack {
            VStack {
                HStack {
                    Text(llm2.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm2.stat)
                }
                HStack {
                    Toggle(isOn: $llm2.includeWeatherTool) {
                        Text("Include \"get current weather\" tool")
                    }
                    .frame(maxWidth: 350, alignment: .leading)
                    Spacer()
                    if llm2.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                }
            }
            // show the model output
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        if selectedDisplayStyle == .plain {
                            Text(llm2.output)
                                .textSelection(.enabled)
                        } else {
                            Markdown(llm2.output)
                                .textSelection(.enabled)
                        }
                    }
                    .onChange(of: llm2.output) { _, _ in
                        sp.scrollTo("bottom")
                    }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                LLMEvaluatorView()
                LLMEvaluator2View()
            }

            HStack {
                TextField("prompt", text: $prompt)
                    .onSubmit(generate)
                    .disabled(llm.running || llm2.running)
                #if os(visionOS)
                    .textFieldStyle(.roundedBorder)
                #endif
                Button(llm.running || llm2.running ? "stop" : "generate", action: llm.running || llm2.running ? cancel : generate)
            }
        }
        #if os(visionOS)
        .padding(40)
        #else
        .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }
        }
        .task {
            // pre-load the weights on launch to speed up the first generation
            _ = try? await llm.load()
            _ = try? await llm2.load()
        }
    }

    private func generate() {
        Task {
            llm.generate(prompt: prompt)
        }
        Task {
            llm2.generate(prompt: prompt)
        }
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

@Observable
@MainActor
class LLMEvaluator {
    var running = false

    var includeWeatherTool = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. `qwen2_5_1_5b` is one of the smaller ones, so this will fit on
    /// more devices.
    let modelConfiguration = LLMRegistry.qwen2_5_1_5b

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.6)
    let maxTokens = 240
    let updateInterval = 0.25

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    let currentWeatherToolSpec: [String: any Sendable] =
        [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        ] as [String: String],
                        "unit": [
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        ] as [String: any Sendable],
                    ] as [String: [String: any Sendable]],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(numParams / (1024 * 1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String) async {
        output = ""
        let userInput = UserInput(prompt: prompt)

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) in
                let lmInput = try await context.processor.prepare(input: userInput)
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context
                )

                var tokenCount = 0
                var lastEmissionTime = Date()
                var chunks = ""

                for await result in stream {
                    switch result {
                    case .chunk(let string):
                        tokenCount += 1
                        if tokenCount >= maxTokens { await generationTask?.cancel() }
                        let now = Date()
                        if now.timeIntervalSince(lastEmissionTime) >= updateInterval {
                            lastEmissionTime = now
                            let text = chunks
                            chunks = ""
                            Task { @MainActor in
                                self.output += text
                            }
                        } else {
                            chunks += string
                        }
                    case .info(let info):
                        Task { @MainActor in
                            self.stat = "\(info.tokensPerSecond) tokens/s"
                        }
                    }
                }

                Task { @MainActor in
                    self.output += chunks
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate(prompt: String) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await generate(prompt: prompt)
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }
}

@Observable
@MainActor
class LLMEvaluator2 {
    var running = false

    var includeWeatherTool = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. `qwen2_5_1_5b` is one of the smaller ones, so this will fit on
    /// more devices.
    let modelConfiguration = LLMRegistry.qwen2_5_0_5b

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.6)
    let maxTokens = 240
    let updateInterval = 0.25

    /// A task responsible for handling the generation process.
    var generationTask: Task<Void, Error>?

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle

    let currentWeatherToolSpec: [String: any Sendable] =
        [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        ] as [String: String],
                        "unit": [
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        ] as [String: any Sendable],
                    ] as [String: [String: any Sendable]],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ] as [String: any Sendable]

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(numParams / (1024 * 1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    private func generate(prompt: String) async {
        output = ""
        let userInput = UserInput(prompt: prompt)

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            try await modelContainer.perform { (context: ModelContext) in
                let lmInput = try await context.processor.prepare(input: userInput)
                let stream = try MLXLMCommon.generate(
                    input: lmInput, parameters: generateParameters, context: context
                )

                var tokenCount = 0
                var lastEmissionTime = Date()
                var chunks = ""

                for await result in stream {
                    switch result {
                    case .chunk(let string):
                        tokenCount += 1
                        if tokenCount >= maxTokens { await generationTask?.cancel() }
                        let now = Date()
                        if now.timeIntervalSince(lastEmissionTime) >= updateInterval {
                            lastEmissionTime = now
                            let text = chunks
                            chunks = ""
                            Task { @MainActor in
                                self.output += text
                            }
                        } else {
                            chunks += string
                        }
                    case .info(let info):
                        Task { @MainActor in
                            self.stat = "\(info.tokensPerSecond) tokens/s"
                        }
                    }
                }

                Task { @MainActor in
                    self.output += chunks
                }
            }

        } catch {
            output = "Failed: \(error)"
        }
    }

    func generate(prompt: String) {
        guard !running else { return }
        generationTask = Task {
            running = true
            await generate(prompt: prompt)
            running = false
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        running = false
    }
}
