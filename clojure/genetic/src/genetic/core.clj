(ns genetic.core)

(defprotocol Chromosome
  (mutate [lhs])
  (recombine [lhs rhs])
  (calc-fitness [lhs]))

(def mrate 0.03)
(def rrate 0.7)

(defn ^{:private true} rate-pred [r _]
  (> r (rand)))

(defn ^{:private true} mutate-pop [mpred pop]
  (map mutate (filter mpred pop)))

(defn ^{:private true} recombine-pop [rpred pop]
  (map #(recombine % (rand-nth pop)) (filter rpred pop)))

(defn ^{:private true} find-unused-index [cnt used]
  (let [i (rand-int cnt)]
    (if (contains? used i)
      (recur cnt used)
      i)))

(defn ^{:private true} choose-by [f pop]
  (reduce #(if (f (calc-fitness %1) (calc-fitness %2)) %1 %2) pop))

(defn ^{:private true} rand-in [l u]
  (let [bound (- u l)]
    (+ l (rand bound))))

(defn ^{:private true} remove-by-index [pop indexes]
  (keep-indexed #(if (contains? indexes %1)
                  nil
                  %2) pop))

(defn ^{:private true} evolve-pop [mutator recombiner selector coll]
  (lazy-seq
    (let [pop (vec coll)
          trans (vec (concat pop (mutator pop) (recombiner pop)))
          new-pop (selector trans (- (count trans) (count pop)))]
      (cons new-pop (evolve-pop mutator recombiner selector new-pop)))))

(defn best [pop]
  (choose-by > pop))

(defn worst [pop]
  (choose-by < pop))

(defn tournament-select [coll n]
  (let [pop (vec coll) cnt (count pop)]
    (loop [dropped #{}, i 0]
      (if (= i n)
        (remove-by-index pop dropped)
        (let [index1 (find-unused-index cnt dropped)
              index2 (find-unused-index cnt dropped)]
          (if (< (calc-fitness (nth pop index1)) (calc-fitness (nth pop index2)))
            (recur (conj dropped index1) (inc i))
            (recur (conj dropped index2) (inc i))))))))

(defn roulette-select [coll n]
  (let [pop (vec coll) cnt (count pop)
        maxf (calc-fitness (best pop))
        minf (calc-fitness (worst pop))]
    (loop [dropped #{}, i 0]
      (if (= i n)
        (remove-by-index pop dropped)
        (let [score (rand-in minf maxf)
              index (find-unused-index cnt dropped)]
          (if (<= (calc-fitness (nth pop index)) score)
            (recur (conj dropped index) (inc i))
            (recur dropped i)))))))

(defn evolve
  ([coll]
    (evolve coll {}))
  ([coll options]
    (let [{:keys [mutationrate recombinationrate selector],
           :or {mutationrate mrate, recombinationrate rrate, selector tournament-select}} options
          mpred (partial rate-pred mutationrate)
          mutator (partial mutate-pop mpred)
          rpred (partial rate-pred recombinationrate)
          recombiner (partial recombine-pop rpred)]
      (evolve-pop mutator recombiner selector coll))))
