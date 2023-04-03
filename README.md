# Process-aware-DNN
A framework combining NLP and DL for performing pitting potential prediction and optimization on electrochemical alloy corrosion data containing numerical and textual input features
Textual inputs are processed through vocabulary tokenization, word embedding followed by LSTM layers to generate data that can be fed to a dense layer.
This processed data is concatenated with numerical inputs including alloy composition, test solution pH, chloride ion concentration etc.
The trained DNN model is also used for composition optimization, by using the gradient descent method. Gradients are calculated using augnet https://github.com/nima-siboni/aug-net
