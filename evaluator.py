import math

class Evaluator:
    def validate_probabilities(self, model):
        epsilon = 0.0001
        for word in model:
            total_probabilities = 0.0
            for next_word in model[word]:
                probability = model[word][next_word]
                if(probability >= 0.0 and probability <= 1.0):
                    total_probabilities += probability
                else:
                    raise ValueError("Probabilities should be between 0 and 1")
            if abs(total_probabilities - 1.0) > epsilon:
                raise ValueError("Probabilities for each word must sum to approximately 1.0")

    def vocabulary_size(self, model) -> int:
        vocab:set[str] = set()
        for word in model:
            vocab.add(word)
            for next_word in model[word]:
                vocab.add(next_word)
        return len(vocab)

    def unique_bigram_count(self, model) -> int:
        unique_bigrams: set[tuple[str,str]] = set()
        for word in model:
            for next_word in model[word]:
                current_bigram = (word, next_word)
                unique_bigrams.add(current_bigram)
        return len(unique_bigrams)
        # improve later by simply counting inner dictionary entries

    def get_bigrams(self, test_tokens: list[str])->list[tuple[str,str]]:
        bigrams = []

        for i in range(len(test_tokens)-1):
            bigrams.append((test_tokens[i],test_tokens[i+1]))

        return bigrams


    def get_probability(
            self,
            model,
            current_word,
            next_word
        )-> float:
        if current_word in model:
            if next_word in model[current_word]:
                return model[current_word][next_word]
        return 0.0

    def evaluate_seen_vs_unseen_bigrams(
        self,
        model,
        test_tokens
    ) -> dict[str,int]:
        test_bigrams = self.get_bigrams(test_tokens)
        seen_bigrams_count = 0
        unseen_bigrams_count = 0
        for first_word, second_word in test_bigrams:
            probability = self.get_probability(model, first_word, second_word)
            if probability > 0.0:
                    seen_bigrams_count += 1
            else:
                unseen_bigrams_count +=1

        return {
            "total_bigrams": len(test_bigrams),
            "seen_bigrams": seen_bigrams_count,
            "unseen_bigrams": unseen_bigrams_count
        }

    def calculate_coverage(
        self,
        seen_bigrams: int,
        total_bigrams: int
    ) -> float:

        if total_bigrams == 0:
            raise ValueError("Total bigrams are zero , cannot calculate coverage")

        return seen_bigrams / total_bigrams
            

    def calculate_log_probabilities(self,model,test_tokens) -> dict[str,float]:
        if len(test_tokens) < 2:
            raise ValueError("Test tokens are less than 2")
        bigrams = self.get_bigrams(test_tokens)
        total_log_probability = 0.0
        for first_word,second_word in bigrams:
            probability = self.get_probability(model,first_word,second_word)
            if probability == 0.0:
                return {
                    "total_log_probability": -math.inf,
                    "average_log_probability" : -math.inf
                } 
            total_log_probability += math.log(probability)
        
        average_log_probability = total_log_probability / len(bigrams)

        return {
            "total_log_probability": total_log_probability,
            "average_log_probability" : average_log_probability
        }

    def calculate_perplexity(self, average_log_probability) -> float:
        if(average_log_probability == -math.inf):
            return math.inf
        return math.exp(-average_log_probability)

    def evaluate(self, model, test_tokens) -> dict[str,int | float]:
        self.validate_probabilities(model)
        log_probabilities = self.calculate_log_probabilities(model,test_tokens)
        bigrams = self.evaluate_seen_vs_unseen_bigrams(model, test_tokens)

        return{
            "vocabulary_size": self.vocabulary_size(model),
            "unique_bigram_count": self.unique_bigram_count(model),
            "evaluated_bigrams": bigrams["total_bigrams"],
            "seen_bigrams": bigrams["seen_bigrams"],
            "unseen_bigrams":bigrams["unseen_bigrams"],
            "coverage": self.calculate_coverage(
                bigrams["seen_bigrams"],
                bigrams["total_bigrams"]
            ),
            "total_log_probability": log_probabilities["total_log_probability"],
            "average_log_probability": log_probabilities["average_log_probability"],
            "perplexity": self.calculate_perplexity(log_probabilities["average_log_probability"])
        }

    def report(self, metrics):
        print("===== Bigram Model Evaluation =====")

        print("Vocabulary size:", metrics["vocabulary_size"])
        print("Unique learned bigrams:", metrics["unique_bigram_count"])

        print("Evaluated bigrams:", metrics["evaluated_bigrams"])
        print("Seen bigrams:", metrics["seen_bigrams"])
        print("Unseen bigrams:", metrics["unseen_bigrams"])

        print("Coverage:", f"{metrics['coverage']:.2%}")

        print("Total log probability:", metrics["total_log_probability"])

        print("Average log probability:", metrics["average_log_probability"])

        print("Perplexity:", metrics["perplexity"])
        

        

    

    

            
