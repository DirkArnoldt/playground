(ns neuron.hopfield
  (:require [neuron.core])
  (:require [util.matrix :as m :refer :all])
  (:import (neuron.core NeuralNet)))

(def ON 1.0)
(def OFF -1.0)

;---

(defn- update-states [threshold weights states]
  (map (fn [r] (if (< (m/dotproduct r states) threshold)
                 OFF
                 ON)) weights))

(defn- recall-state [threshold max-iter weights pattern]
  (loop [previous nil
         states pattern
         i 0]
    (if (or (= i max-iter) (= states previous))
      states
      (recur states (update-states threshold weights states) (inc i)))))

;##### Hebbian learning rule #####

(defn- calc-hebbian-weight [patterns _ [i j]]
  (let [n (count patterns)
        sum (reduce + 0.0 (map #(* (get % i) (get % j)) patterns))]
    (/ sum n)))

(defn- train-hebbian [weights patterns]
  (let [f (partial calc-hebbian-weight patterns)]
    (m/transform-with-index f weights)))

;##### Storkey learning rule #####

(defn- calc-storkey-height [weights pattern _ [i j]]
  (let [n (count pattern)]
    (reduce + 0.0 (map #(* (get-in weights [i %]) (get pattern %))
                       (filter #(and (not (= i %)) (not (= j %))) (range n))))))

(defn- calc-storkey-heights [weights pattern]
  (let [size (m/column-count weights)
        heights (m/initialize-matrix [size size] 0.0)
        f (partial calc-storkey-height weights pattern)]
    (m/transform-with-index f heights)))

(defn- calc-storkey-weight [n heights pattern v [i j]]
  (let [term1 (* (/ 1 n) (pattern i) (pattern j))
        term2 (* (/ -1 n) (pattern i) (get-in heights [j i]))
        term3 (* (/ -1 n) (pattern j) (get-in heights [i j]))]
    (+ v term1 term2 term3)))

(defn- train-storkey-pattern [weights pattern]
  (let [n (count pattern)
        heights (calc-storkey-heights weights pattern)
        f (partial calc-storkey-weight n heights pattern)]
    (m/transform-with-index f weights)))

(defn- train-storkey [weights patterns]
  (loop [weights weights
         patterns patterns]
    (if (seq patterns)
      (recur (train-storkey-pattern weights (first patterns)) (rest patterns))
      weights)))

;--- helpers

(defn- pick-lrule-fn [lrule]
  (cond
    (= lrule :hebbian) train-hebbian
    (= lrule :storkey) train-storkey
    :else (throw (IllegalArgumentException. (str "unknown learning rule " lrule)))))

;--- public API ---

(defrecord Hopfield [threshold max-iterations f-learn weights]
  NeuralNet
  (train [_ patterns]
    (Hopfield. threshold max-iterations f-learn (f-learn weights patterns)))
  (recall [_ pattern]
    (recall-state threshold max-iterations weights pattern)))

(defn energy [net states]
  (let [weights (:weights net)
        size (count states)
        term1 (reduce + 0.0 (map #(* (:threshold net) %) states))
        term2 (reduce + 0.0 (for [i (range size) j (range size)]
                              (* (get-in weights [i j]) (states i) (states j))))]
    (+ term1 (* -0.5 term2))))

(defn build [size options]
  (let [{:keys [lrule threshold max-iterations],
         :or { lrule :hebbian, threshold 0.0, max-iterations 3}} options
        weights (initialize-matrix [size size] 0.0)
        f-learn (pick-lrule-fn lrule)]
    (Hopfield. threshold max-iterations f-learn weights)))

