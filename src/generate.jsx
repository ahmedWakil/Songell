import * as ort from "onnxruntime-web";

function encodeCategory(categoryIndex, categories) {
    const K = categories.length;
    let categoryArray = new Array(K).fill(0);
    categoryArray[categoryIndex] = 1;
    const dims = [1, K];

    const categoryTensor = new ort.Tensor("float32", categoryArray, dims)
    return categoryTensor;
}

function encodeInput(charIndex, char_set) {
    const N = char_set.length;
    let inputArray = new Array(N).fill(0);
    inputArray[charIndex] = 1;
    const dims = [1, N];

    const inputTensor = new ort.Tensor("float32", inputArray, dims);
    return inputTensor;
}

function initHidden(size) {
    let hiddenArray = new Array(size).fill(0);
    const dims = [1, size];

    const h0 = new ort.Tensor("float32", hiddenArray, dims);
    return h0;
}

function collectFeed(inputNames, inputList) {
    const feed = {};

    inputNames.map((inputName, index) => {
        feed[inputName] = inputList[index];
    })

    return feed;
}

function topK(logits, K) {
    const orderedLogits = [...logits.data].sort((a, b) => b - a);
    const topLogit = orderedLogits[Math.floor(Math.random() * K)];
    return logits.data.indexOf(topLogit);
}

export async function sample(categoryIndex, modelData, maxLength = 40) {
    // create a infrencing session
    const session = await ort.InferenceSession.create('./infrencing-model/learned-weights-onnx.onnx');
    console.log(`the: ${ort.env.wasm.wasmPaths}`)
    // initial preperations for sampling
    const sosi = modelData.char_set.indexOf(modelData.sos);
    const eosi = modelData.char_set.indexOf(modelData.eos);
    const categoryTensor = encodeCategory(categoryIndex, modelData.categories);
    let inputTensor = encodeInput(sosi, modelData.char_set);
    let hn = initHidden(modelData.h_size);
    let cn = initHidden(modelData.h_size);

    let sample = "";
    let letterIndex = sosi;
    while (sample.length <= maxLength) {
        // first do a forward pass
        const feed = collectFeed(session.inputNames, [categoryTensor, inputTensor, hn, cn]);
        const outputMap = await session.run(feed);
        // choose a random letter from the top K largest class probabilities
        if (sample.length === 0) {
            letterIndex = topK(outputMap.output, 10);
        } else {
            letterIndex = topK(outputMap.output, 2);
        }
        // if the model sampled a eos we can stop evaluating for new characters
        if (letterIndex === eosi) {
            break;
        }
        sample += modelData.char_set[letterIndex];
        // update the inputs
        inputTensor = encodeInput(letterIndex, modelData.char_set);
        hn = outputMap.hiddenOUT;
        cn = outputMap.cellOUT;
    }

    return sample
}