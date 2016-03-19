(ns genetic.core)

(def MUTATION_RATE 0.03)
(def RECOMBINATION_RATE 0.7)

(defprotocol Chromosome
  (mutate [lhs])
  (recombine [lhs rhs])
  (fitness [lhs]))


(defn- rate-pred [r _]
  (> r (rand)))

(defn- find-unused-index [size used]
  (let [i (rand-int size)]
    (if (contains? used i)
      (recur size used)
      i)))

(defn- choose-by [f pop]
  (reduce #(if (f (fitness %1) (fitness %2)) %1 %2) pop))

(defn- rand-in [l u]
  (let [bound (- u l)]
    (+ l (rand bound))))

(defn- remove-indexes [pop indexes]
  (keep-indexed #(if (contains? indexes %1)
                  nil
                  %2) pop))


(defn- mutate-pop [mpred pop]
  (map mutate (filter mpred pop)))

(defn- recombine-pop [rpred pop]
  (map #(recombine % (rand-nth pop)) (filter rpred pop)))

(defn- evolve-pop [mutator recombiner selector coll]
  (lazy-seq
    (let [pop (vec coll)
          trans (vec (concat pop (mutator pop) (recombiner pop)))
          new-pop (selector (count pop) trans )]
      (cons new-pop (evolve-pop mutator recombiner selector new-pop)))))


(defn best [pop]
  (choose-by > pop))

(defn worst [pop]
  (choose-by < pop))

(defn tournament-select [n coll]
  (let [pop (vec coll)
        cnt (count pop)]
    (loop [dropped #{}, i cnt]
      (if (= i n)
        (remove-indexes pop dropped)
        (let [index1 (find-unused-index cnt dropped)
              index2 (find-unused-index cnt dropped)]
          (if (< (fitness (nth pop index1)) (fitness (nth pop index2)))
            (recur (conj dropped index1) (dec i))
            (recur (conj dropped index2) (dec i))))))))

(defn roulette-select [n coll]
  (let [pop (vec coll)
        cnt (count pop)
        maxf (fitness (best pop))
        minf (fitness (worst pop))]
    (loop [dropped #{}, i cnt]
      (if (= i n)
        (remove-indexes pop dropped)
        (let [score (rand-in minf maxf)
              index (find-unused-index cnt dropped)]
          (if (<= (fitness (nth pop index)) score)
            (recur (conj dropped index) (dec i))
            (recur dropped i)))))))

(defn evolve
  ([coll]
    (evolve coll {}))
  ([coll options]
    (let [{:keys [mutationrate recombinationrate selector],
           :or {mutationrate MUTATION_RATE, recombinationrate RECOMBINATION_RATE, selector tournament-select}} options
          mpred (partial rate-pred mutationrate)
          mutator (partial mutate-pop mpred)
          rpred (partial rate-pred recombinationrate)
          recombiner (partial recombine-pop rpred)]
      (evolve-pop mutator recombiner selector coll))))
