(ns regression.core
  (:require [util.matrix :as m]
            [util.regularization :as reg]
            [util.cost :as cost]))

(declare sum-error)

(defn feature-size [training-set]
  (count (first (first training-set))))

(defn l2reg-cost [lambda theta]
  (* lambda (reduce + 0.0 (map #(* %1 %1) theta))))

(defn add [a b]
  (if (coll? a)
    (m/add a b)
    (+ a b)))

(defn calc-hypothesis [hypothesis sample]
  (let [x (first sample)
        y (second sample)
        hx (hypothesis x)]
    [hx y]))

(defn simple-error [hypothesis sample]
  (let [[hx y] (calc-hypothesis hypothesis sample)]
    (cost/error hx y)))

(defn error-derivate-item [hypothesis sample]
  (let [x (first sample)
        e (simple-error hypothesis sample)]
    (mapv #(* e %1) x)))

(defn error-derivate [hypothesis training-set]
  (let [acc (m/zero-vector (feature-size training-set))]
    (sum-error acc error-derivate-item hypothesis training-set)))

(defn sum-error [acc ef hypothesis training-set]
  (loop [eacc acc
         training-set training-set
         m 0]
    (if (seq training-set)
      (recur (add eacc (ef hypothesis (first training-set))) (rest training-set) (inc m))
      [eacc m])))

(defn gradient-descent [cf' alpha lambda hypothesis training-set theta]
  (let [[errors m] (cf' hypothesis training-set)
        lrate (/ alpha m)
        rescale (reg/l2-regularization lrate lambda)]
    (mapv #(- (rescale %1) (* lrate %2)) theta errors)))


