import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const upstream = process.argv[2];
const load = (relative) => import(pathToFileURL(path.join(upstream, relative)).href);
const {
  parseZCodeBuiltinProviderConfigRules,
  parsePersonalProviderConfigMap,
  parseZCodeBuiltinModelConfigRules,
  parsePersonalModelConfigRules,
  ProviderConfigMap,
  ProviderConfigResolver,
} = await load("packages/provider/src/index.ts");
const { AiSdkModelAdapter } = await load(
  "apps/zcode-cli/packages/adapters/src/model/runner.ts",
);
const { resolveNormalRequestMaxOutputTokens, resolveModelStepMaxOutputTokens } = await load(
  "apps/zcode-cli/packages/core/src/runtime/methods/model-token-limits.ts",
);
let inputText = "";
for await (const chunk of process.stdin) inputText += chunk;
const input = JSON.parse(inputText);
const reports = [];

function resolveModel(builtinEnvelope, personalEnvelope) {
  const builtin = builtinEnvelope.config;
  const personal = personalEnvelope.config;
  const providers = parseZCodeBuiltinProviderConfigRules(builtin.providerConfigRules);
  const resolution = new ProviderConfigResolver().resolve({
    zcodeBuiltinProviders: providers.providers,
    zcodeBuiltinProviderTemplates: providers.providerTemplates,
    personalProviders: parsePersonalProviderConfigMap(personal.providerConfigRules),
    zcodeBuiltinModelRules: parseZCodeBuiltinModelConfigRules(builtin.modelConfigRules),
    personalModels: parsePersonalModelConfigRules(personal.modelConfigRules),
    accountProviders: new ProviderConfigMap(),
    personalProviderOrder: personal.providerOrder,
  });
  const selection = personal.defaultModelSelection;
  const provider = resolution.registryProviders.find(
    (candidate) => candidate.providerId === selection.providerId,
  );
  assert.ok(provider, JSON.stringify(resolution.issues));
  const model = provider.models.find((candidate) => candidate.modelId === selection.modelId);
  assert.ok(model, JSON.stringify(resolution.issues));
  return { provider, model };
}

for (const testCase of input.cases) {
  const baseline = resolveModel(input.upstreamBuiltin, testCase.personal);
  const { provider, model } = resolveModel(testCase.builtin, testCase.personal);
  assert.equal(provider.providerId, "zai");
  assert.equal(provider.config.group, "standard-personal");
  const override = testCase.builtin.config.modelConfigRules.builtinProviderModelRules.at(-1);
  if (testCase.override === null) {
    assert.ok(!Object.hasOwn(override.config.optionSpecs, "maxOutputTokens"));
    assert.deepEqual(
      model.config.optionSpecs.maxOutputTokens,
      baseline.model.config.optionSpecs.maxOutputTokens,
    );
  } else {
    assert.equal(model.config.optionSpecs.maxOutputTokens.max, testCase.override);
  }
  assert.equal(model.config.properties.contextWindow, testCase.contextWindow);
  const effectiveMaxOutputTokens = resolveModelStepMaxOutputTokens({
    baselineMaxOutputTokens: resolveNormalRequestMaxOutputTokens({
      modelMaxOutputTokens: model.config.optionSpecs.maxOutputTokens.max,
    }),
    contextWindow: model.config.properties.contextWindow,
    estimatedCurrentUsage: 100,
    modelContextBudgetStrategy: "preflight-v1",
  });
  const requests = [];
  globalThis.fetch = async (request, init) => {
    const body = init?.body ?? (request instanceof Request ? await request.text() : undefined);
    requests.push(JSON.parse(String(body)));
    return new Response(JSON.stringify({
      id: "mock-response",
      object: "chat.completion",
      created: 1,
      model: testCase.model,
      choices: [{ index: 0, message: { role: "assistant", content: "done" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }), { headers: { "content-type": "application/json" } });
  };
  const adapter = new AiSdkModelAdapter({ env: {}, retry: { maxAttempts: 1 } });
  const executable = adapter.createModel({
    providerId: provider.providerId,
    modelId: model.modelId,
    providerConfig: provider.config,
    modelConfig: model.config,
  });
  await executable.generateText({
    messages: [{ role: "user", content: "hello" }],
    options: { reasoningLevel: "enabled", maxOutputTokens: effectiveMaxOutputTokens },
  });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].max_completion_tokens, effectiveMaxOutputTokens);
  reports.push({
    model: testCase.model,
    override: testCase.override,
    nativeMaxOutputTokens: baseline.model.config.optionSpecs.maxOutputTokens.max,
    effectiveMaxOutputTokens,
    request: requests[0],
  });
}
console.log(JSON.stringify(reports));
