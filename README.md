# Process-aware-DNN
A framework combining NLP and DL for performing pitting potential prediction and optimization on electrochemical alloy corrosion data containing numerical and textual input features
Textual inputs are processed through vocabulary tokenization, word embedding followed by LSTM layers to generate data that can be fed to a dense layer.
This processed data is concatenated with numerical inputs including alloy composition, test solution pH, chloride ion concentration etc.
The trained DNN model is also used for composition optimization, by using the gradient descent method. Gradients are calculated using augnet https://github.com/nima-siboni/aug-net

The code has three main components: 1) Building the NLP+DL model capable of being trained on both numerical and textual data; 2) Creating so-called sub-models, i.e. breaking the original model at the concatenation layer. Post-concatenation sub-model is used for deriving gradients for optimization. This had to be done since the data-structure prior to concatenation does not allow calculation of gradients. Sub-models are created by deriving the final trained weights of the original model, for all respective layers. 3) Composition optimization
