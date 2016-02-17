(ns genetic.example
  (:require [genetic.core :as g]
            [clojure.string :as string]
            [clojure.tools.cli :refer [parse-opts]])
  (:import (genetic.core Chromosome))
  (:gen-class))

(defn generate-value []
  (repeatedly 10 #(rand-int 256)))

(def target (generate-value))

(defn calc-fitness [v]
  (Math/abs (- 1.0 (/ (reduce + (mapv #(Math/abs (int (- %1 %2))) v target)) 2550))))

(defrecord NumChromosome [value fvalue]
  Chromosome
  (mutate [_]
    (let [split (rand-int 9)
          v (concat (take split value) (vector (rand-int 256)) (drop (+ split 1) value))]
      (NumChromosome. v (calc-fitness v))))
  (recombine [_ rhs]
    (let [split (rand-int 10)
          v (concat (take split value) (drop split (:value rhs)))]
      (NumChromosome. v (calc-fitness v))))
  (fitness [_]
    fvalue))

(defn make-chromosome []
  (let [v (generate-value)]
    (NumChromosome. v (calc-fitness v))))

(defn make-population
  ([n]
   (repeatedly n make-chromosome))
  ([]
   (repeatedly 100 make-chromosome)))



(def selector-map {"tournament" g/tournament-select,
                   "roulette" g/roulette-select})

(def cli-options
  [
   ["-p" "--population-size SIZE" "Population size"
    :default 100
    :parse-fn #(Integer/parseInt %)]
   ["-g" "--generation-count COUNT" "Generation count"
    :default 50
    :parse-fn #(Integer/parseInt %)]
   ["-m" "--mutationrate RATE" "Mutation rate"
    :default 0.05
    :parse-fn #(Float/parseFloat %)
    :validate [#(< 0.0 % 1.0) "Must be a floating point number between 0.0 and 1.0"]]
   ["-r" "--recombinationrate RATE" "Recombination rate"
    :default 0.7
    :parse-fn #(Float/parseFloat %)
    :validate [#(< 0.0 % 1.0) "Must be a floating point number between 0.0 and 1.0"]]
   ["-s" "--selector SELECTOR" "Selection strategy"
    :default g/tournament-select
    :parse-fn #(selector-map %)
    :validate [#(not (nil? %)) "Must be one of ['tournament', 'roulette']"]]
   ["-h" "--help"]])

(defn usage [options-summary]
  (->> ["This is a sample program for the genetic algorithm."
        ""
        "Usage: program-name [options]"
        ""
        "Options:"
        options-summary]
       (string/join \newline)))

(defn error-msg [errors]
  (str "The following errors occurred while parsing your command:\n\n"
       (string/join \newline errors)))

(defn exit [status msg]
  (println msg)
  (System/exit status))

(defn -main [& args]
  (let [{:keys [options arguments errors summary]} (parse-opts args cli-options)]
    ;; Handle help and error conditions
    (cond
      (:help options) (exit 0 (usage summary))
      errors (exit 1 (error-msg errors)))
    (let [population (make-population (:population-size options))
          generations (:generation-count options)
          best (g/best (last (take generations (g/evolve population options))))]
      (do
        (println target)
        (print (:value best))
        (print " / fitness: ")
        (println (:fvalue best))))))