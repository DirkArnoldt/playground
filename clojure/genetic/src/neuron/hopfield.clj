(ns neuron.hopfield
  (:require [neuron.core])
  (:import (neuron.core NeuralNet)))

(def ON 1.0)
(def OFF -1.0)

(def ^:dynamic *treshold* 0.1)
(def ^:dynamic *iterations* 1)

(defn- initialize-matrix [size]
  (vec (map vec (partition size (repeat (* size size) 0.0)))))

;---

(defn- update-state [weights states index]
  (let [temp (reduce + 0.0 (map-indexed #(* (get-in weights [index %1]) %2) states))]
    (if (< temp *treshold*)
      (assoc states index OFF)
      (assoc states index ON))))

(defn- update-states [weights states]
  (let [size (count states)]
    (loop [states states
           index 0]
      (if (= index size)
        states
        (recur (update-state weights states index) (inc index) )))))

(defn- recall-state [weights pattern]
  (loop [states pattern
         iterations *iterations*]
    (if (= 0 iterations)
      states
      (recur (update-states weights states) (dec iterations)))))

;----

(defn- calc-hebbian-weight [i j patterns]
  (let [n (count patterns)
        sum (reduce + 0.0 (map #(* (get % i) (get % j)) patterns))]
    (/ sum n)))

(defn- train-hebbian-patterns [weights patterns i]
  (loop [weights weights
         j 0]
    (if (= j i)
      weights
      (let [weight (calc-hebbian-weight i j patterns)
            new-weights (assoc-in (assoc-in weights [i j] weight) [j i] weight)]
        (recur new-weights (inc j))))))

(defn- train-hebbian [weights patterns]
  (let [size (count (weights 0))]
    (loop [weights weights
           i 0]
      (if (= i size)
        weights
        (recur (train-hebbian-patterns weights patterns i) (inc i))))))

;---

(defn- calc-storkey-height [weights i j pattern]
  (let [n (count pattern)]
    (reduce + 0.0 (map #(* (get-in weights [i %]) (get pattern %))
                       (filter #(and (not (= i %)) (not (= j %))) (range n))))))

(defn- calc-heights1 [size weights pattern heights i]
  (loop [heights heights
         j 0]
    (if (= j size)
      heights
      (let [height (calc-storkey-height weights i j pattern)
            new-heights (assoc-in heights [i j] height)]
        (recur new-heights (inc j))))))

(defn- calc-heights [weights pattern]
  (let [size (count (weights 0))]
    (loop [heights (initialize-matrix size)
           i 0]
      (if (= i size)
        heights
        (recur (calc-heights1 size weights pattern heights i) (inc i))))))

;---

(defn calc-storkey-weight [n heights weights i j pattern]
  (let [term1 (* (/ 1 n) (pattern i) (pattern j))
        term2 (* (/ -1 n) (pattern i) (get-in heights [j i]))
        term3 (* (/ -1 n) (pattern j) (get-in heights [i j]))]
    (+ (get-in weights [i j]) term1 term2 term3)))

(defn calc-storkey-weights [size n heights weights i pattern]
  (loop [weights weights
         j 0]
    (if (= j size)
      weights
      (let [weight (calc-storkey-weight n heights weights i j pattern)
            new-weights (assoc-in weights [i j] weight)]
        (recur new-weights (inc j))))))

(defn- train-storkey-pattern [weights pattern]
  (let [n (count pattern)
        heights (calc-heights weights pattern)
        size (count (weights 0))]
    (loop [weights weights
           i 0]
      (if (= i size)
        weights
        (recur (calc-storkey-weights size n heights weights i pattern) (inc i))))))

(defn- train-storkey [weights patterns]
  (loop [weights weights
         patterns patterns]
    (if (seq patterns)
      (recur (train-storkey-pattern weights (first patterns)) (rest patterns))
      weights)))

;---

(defn- pick-lrule-fn [lrule]
  (cond
    (= lrule :hebbian) train-hebbian
    (= lrule :storkey) train-storkey
    :else (throw (IllegalArgumentException. (str "unknown learning rule " lrule)))))

;--- public API ---

(defrecord Hopfield [treshold f weights]
  NeuralNet
  (train [_ patterns]
    (Hopfield. treshold f (f weights patterns)))
  (recall [_ pattern]
    (recall-state weights pattern)))

(defn energy [net states]
  (let [weights (:weights net)
        size (count states)
        term1 (reduce + 0.0 (map #(* *treshold* %) states))
        term2 (reduce + 0.0 (for [i (range size) j (range size)]
                              (* (get-in weights [i j]) (states i) (states j))))]
    (+ term1 (* -0.5 term2))))

(defn build [size lrule]
  (let [weights (initialize-matrix size)
        lrule-fn (pick-lrule-fn lrule)]
    (Hopfield. *treshold* lrule-fn weights)))
