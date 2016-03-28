(ns regression.logistic
  (:require [regression.core :refer :all])
  (:require [util.matrix :as m]
            [util.random :as rnd]
            [util.cost :as cost]
            [util.activation :as a]))

(defn- hf [theta x]
  (a/sigmoid (m/dot theta x)))

(defn- log-likelihood [hypothesis sample]
  (let [[hx y] (calc-hypothesis hypothesis sample)]
    (cost/log-likelihood hx y)))

(defn log-likelihood-cost [lambda theta training-set]
  (let [hypothesis (partial hf theta)
        [error m] (sum-error 0.0 log-likelihood hypothesis training-set)
        ereg (l2reg-cost lambda theta)]
    (+ (* -1 (/ error m)) (/ ereg (* 2 m)))))

(def log-likelihood' error-derivate)

(defn logistic-regression [alpha lambda iterations training-set]
  (let [theta (m/compute-vector (feature-size training-set) rnd/rand-gaussian)]
    (loop [theta theta
           epoche 0]
      (if (= epoche iterations)
        theta
        (let [hypothesis (partial hf theta)]
;          (print theta) (print " : ") (println (log-likelihood-cost lambda theta hypothesis training-set))
          (recur (gradient-descent log-likelihood' alpha lambda hypothesis training-set theta) (inc epoche)))))))

