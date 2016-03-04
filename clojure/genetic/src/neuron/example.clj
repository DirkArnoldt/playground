(ns neuron.example
  (:require [neuron.core :as n])
  (:require [neuron.hopfield :as h]))

(def digits [
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             ])

(defn -main [& args]
  (time
    (let [n (h/build 35 :hebbian)
          n' (n/train n digits)]
      (doall
        (for [i (range 10)]
          (do
          (println (digits i))
          (println (n/recall n' (digits i)))
          (println "---")))))))
