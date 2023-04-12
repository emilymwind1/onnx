using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

namespace onnx.Models
{
    public class OnnxModelScorer : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int[] _inputShape;
        private readonly int[] _outputShape;

        public OnnxModelScorer(string modelPath)
        {
            _session = new InferenceSession(modelPath);

            // Get the input and output metadata
            _inputName = _session.InputMetadata.Keys.First();
            _inputShape = _session.InputMetadata[_inputName].Dimensions.ToArray();
            var outputName = _session.OutputMetadata.Keys.First();
            _outputShape = _session.OutputMetadata[outputName].Dimensions.ToArray();
        }

        public float[] Score(float[] inputs)
        {
            // Create the input tensor
            var tensor = new DenseTensor<float>(new[] { 1, _inputShape[1] }, inputs);

            // Run the prediction
            var inputsList = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>(_inputName, tensor) };
            using (var results = _session.Run(inputsList))
            {
                // Get the output tensor
                var outputTensor = results.First().AsTensor<float>();

                // Extract the output values
                float[] outputValues = new float[_outputShape[1]];
                for (int i = 0; i < _outputShape[1]; i++)
                {
                    outputValues[i] = outputTensor[0, i];
                }

                return outputValues;
            }
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
