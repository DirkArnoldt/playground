(ns neuron.mlp
  (:require [neuron.core])
  (:require [util.matrix :as m])
  (:require [util.activation :as a])
  (:import (neuron.core NeuralNet)))

(def LEARNING-RATE 0.75)
(def MOMENTUM 0.25)

;---

(defn- calc-j-net-input [input j-weights]
  (reduce + 0.0 (mapv * input j-weights)))

(defn- calc-net-inputs [weights input]
  (let [osize (count (weights 0))]
    (vec
      (for [j (range osize)]
        (calc-j-net-input input (m/column weights j))))))

(defn- calc-outputs [af net-input]
  (vec (map #(af %) net-input)))

(defn- calc-output-errors [output target]
  (vec (mapv - target output)))

(defn- calc-j-error [j-weights deltas]
  (reduce + 0.0 (mapv * j-weights deltas)))

(defn- calc-hidden-errors [weights deltas]
  (vec (map #(calc-j-error % deltas) weights)))

(defn- calc-deltas [af' net-inputs errors]
  (vec (map-indexed (fn [j e] (* (af' (net-inputs j)) e)) errors)))

(defn- calc-weight-deltas [lr input deltas momentum dweights]
  (let [si (count input)
        sj (count deltas)
        weight-deltas (for [i (range si) j (range sj)]
                        (+ (* lr (deltas j) (input i))
                           (* momentum (get-in dweights [i j]))))]
    (m/to-matrix sj weight-deltas)))

(defn- update-weights [weights deltas]
  (m/transform-with-index (fn [v idx] (+ v (get-in deltas idx)) ) weights))

;---

(defprotocol Layer
  (train-layer [this input target])
  (feed [this input]))

(defrecord HiddenLayer [weights dweights options next]
  Layer
  (train-layer [_ input target]
    (let [af (:af options)
          af' (:af' options)
          lrate (:lrate options)
          momentum (:momentum options)
          net-inputs (calc-net-inputs weights input)
          output (calc-outputs af net-inputs)
          [layer errors] (train-layer next output target)
          deltas (calc-deltas af' net-inputs errors)
          back-errors (calc-hidden-errors weights deltas)
          weight-deltas (calc-weight-deltas lrate input deltas momentum dweights)
          new-weights (update-weights weights weight-deltas)]
      [(HiddenLayer. new-weights weight-deltas options layer) back-errors]))
  (feed [_ input]
    (let [af (:af options)
          net-inputs (calc-net-inputs weights input)]
      (feed next (calc-outputs af net-inputs)))))

(defrecord OutputLayer [weights dweights options]
  Layer
  (train-layer [_ input target]
    (let [af (:af options)
          af' (:af' options)
          lrate (:lrate options)
          momentum (:momentum options)
          net-inputs (calc-net-inputs weights input)
          output (calc-outputs af net-inputs)
          errors (calc-output-errors output target)
          deltas (calc-deltas af' net-inputs errors)
          back-errors (calc-hidden-errors weights deltas)
          weight-deltas (calc-weight-deltas lrate input deltas momentum dweights)
          new-weights (update-weights weights weight-deltas)]
      [(OutputLayer. new-weights weight-deltas options) back-errors]))
  (feed [_ input]
    (let [af (:af options)
          net-inputs (calc-net-inputs weights input)]
      (calc-outputs af net-inputs))))

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

(defn- build-layer [coll options]
  (let [[i j & remain] coll
        dim (vector i j)
        weights (m/initialize-matrix2 dim rand)
        dweights (m/initialize-matrix dim 0.0)]
    (if (seq remain)
      (let [next (build-layer (rest coll) options)]
        (HiddenLayer. weights dweights options next))
      (OutputLayer. weights dweights options))))

(defn build [coll options]
  (let [{:keys [lrate momentum af af'],
         :or {lrate LEARNING-RATE, momentum MOMENTUM, af a/tanh af' a/tanh'}} options
        layer-options {:lrate lrate, :momentum momentum, :af af, :af' af'}]
    (MultiLayerPerceptron. (build-layer coll layer-options))))

(defn error [target output]
  (* 0.5 (reduce + 0.0 (mapv #(Math/pow (- %1 %2) 2) target output))))