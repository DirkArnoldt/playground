(ns neuron.core)

(defprotocol NeuralNet
  (train [this patterns])
  (recall [this pattern]))

