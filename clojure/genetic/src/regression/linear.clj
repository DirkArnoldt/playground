(ns regression.linear
  (:require [regression.core :refer :all])
  (:require [util.matrix :as m]
            [util.cost :as cost]))
(defn- hf [theta x]
  (m/dot theta x))

(defn- squared-error [hypothesis sample]
  (let [[hx y] (calc-hypothesis hypothesis sample)]
    (cost/squared-error hx y)))

(defn squared-error-cost [lambda theta training-set]
  (let [hypothesis (partial hf theta)
        [error m] (sum-error 0.0 squared-error hypothesis training-set)
        ereg (l2reg-cost lambda theta)]
    (/ (+ error ereg) (* 2 m))))

(def squared-error' error-derivate)

(defn linear-regression [alpha lambda iterations training-set]
  (let [theta (m/zero-vector (feature-size training-set))]
    (loop [theta theta
           epoche 0]
      (if (= epoche iterations)
        theta
        (let [hypothesis (partial hf theta)]
;          (print theta) (print " : ") (println (squared-error-cost lambda theta hypothesis training-set))
          (recur (gradient-descent squared-error' alpha lambda hypothesis training-set theta) (inc epoche)))))))
