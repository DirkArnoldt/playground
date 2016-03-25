(ns neuron.mlp
  (:require [neuron.core])
  (:require [util.matrix :as m])
  (:require [util.activation :as a])
  (:require [clojure.core.matrix :as mm])
  (:require [clojure.core.matrix.random :as mr])
  (:import (neuron.core NeuralNet)))

(def LEARNING-RATE 0.025)
(def MOMENTUM 0.25)
(def REGULARIZATION-PARAM 5.0)

(defprotocol Layer
  (train-layer [this input target])
  (feed [this input]))

;---

(defn calc-activations [af weights bias input]
  (let [z (mm/add (mm/mmul input weights) bias)
        a (mm/emap af z)]
    [z a]))

(defn- calc-hidden-errors [weights deltas]
  (mm/mmul weights deltas))

(defn- calc-output-deltas [af' net-inputs errors]
  (mm/mul (mm/emap af' net-inputs) errors))

(defn- calc-output-errors [output target]
  (mm/sub output target))

(defn- costf-quadratic [af' net-inputs output target]
  (let [errors (calc-output-errors output target)]
    (calc-output-deltas af' net-inputs errors)))

(defn- costf [af' net-inputs output target]
  (let [errors (calc-output-errors output target)]
    errors))

(defn- calc-deltas [next af' net-inputs output target]
  (if (nil? next)
    (vector nil (costf af' net-inputs output target))
    (let [[layer errors] (train-layer next output target)
          deltas (calc-output-deltas af' net-inputs errors)]
      (vector layer deltas))))

(defn- calc-weight-deltas [lr inputs deltas momentum dweights]
  (mm/add (mm/mul lr (mm/outer-product inputs deltas)) (mm/mul momentum dweights)))

(defn- calc-bias-deltas [lr deltas momentum dbias]
  (mm/add (mm/mul lr deltas) (mm/mul momentum dbias)))

(defn- update-weights [weights deltas rf]
  (mm/sub (mm/emap rf weights) deltas))

(defn- update-bias [bias deltas]
  (mm/sub bias deltas))

;---

(defn- feedforward [layer input target]
  (let [options (:options layer)
        af (:af options)
        af' (:af' options)
        weights (:weights layer)
        bias (:bias layer)
        [net-inputs output] (calc-activations af weights bias input)]
    (calc-deltas (:next layer) af' net-inputs output target)))

(defrecord GLayer [weights dweights bias dbias options next]
  Layer
  (train-layer [this input target]
    (let [lrate (:lrate options)
          momentum (:momentum options)
          rf (:rf options)
          ; feedforward
          [layer deltas] (feedforward this input target)
          ; backpropagate
          back-errors (calc-hidden-errors weights deltas)
          weight-deltas (calc-weight-deltas lrate input deltas momentum dweights)
          bias-deltas (calc-bias-deltas lrate deltas momentum dbias)
          new-weights (update-weights weights weight-deltas rf)
          new-bias (update-bias bias bias-deltas)]
      [(GLayer. new-weights weight-deltas new-bias bias-deltas options layer) back-errors]))
  (feed [_ input]
    (let [af (:af options)
          [_ outputs] (calc-activations af weights bias input)]
      (if (nil? next)
        outputs
        (feed next outputs)))))


(defrecord MultiLayerPerceptron [layer]
  NeuralNet
  (train [_ trainings-set]
    (loop [trainings-set trainings-set
           current layer]
      (if (seq trainings-set)
        (let [[input target] (first trainings-set)]
          (recur (rest trainings-set) (first (train-layer current input target))))
        (MultiLayerPerceptron. current))))
  (recall [_ pattern]
    (feed layer pattern)))

(defn- randit [x]
  (fn [] (a/rand-gaussian 0 x)))

(defn- build-layer [coll options]
  (let [[i j & remain] coll
        dim (vector i j)
        bias (vec (take j (repeatedly a/rand-gaussian)))
        dbias (vec (take j (repeat 0.0)))
        weights (m/initialize-matrix2 dim (randit (/ 1.0 (Math/sqrt i))))
        dweights (m/initialize-matrix dim 0.0)]
    (if (seq remain)
      (let [next (build-layer (rest coll) options)]
        (GLayer. weights dweights bias dbias options next))
      (GLayer. weights dweights bias dbias options nil))))

(defn build [coll options]
  (let [{:keys [lrate momentum af af' rf],
         :or {lrate LEARNING-RATE, momentum MOMENTUM, af a/sigmoid, af' a/sigmoid', rf identity}} options
        layer-options {:lrate lrate, :momentum momentum, :af af, :af' af', :rf rf}]
    (MultiLayerPerceptron. (build-layer coll layer-options))))

