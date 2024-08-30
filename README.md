# Running our method with `execute.py`

To execute, configure the following parameters:

1. **`data`**: `string`
   - the name of the data, such as `'agnews'`, `'SST-2'`, `'sst-5'`, note that it needs to have the same name as the dataset.

2. **`method`**: `string`
   - The type of method for selecting examples:
   - - `'Default'` for no example
     - `'Random'` for 10 random examples
     - `'RandomByCategory'` for 2 random examples per each category
     - `'LENS'` examples selected using LENS method, see https://arxiv.org/pdf/2302.13539 for more details

3. **`client`**: `OpenAI object`
   - Your OpenAI API key and base URL.

4. **`model`**: `str`
   - an object string for llm models such as `'deepseek-chat'`.

5. **`batch_size`**: `int`
   - number of data given to the llm for prediction each time.

---
### The following parameters only apply to LENS method:
1. **`valid_size`**: `int`
   - validation data size used during diversity guided search.

2. **`data_size`**: `int`
   - the size of the data used for progressive example filtering, a large number may cause the process to take a long time.

3. **`progressive_factor`**: `int`
   - progressive factor `p`, see original paper for more detail.

4. **`desired_candidate_size`**: `int`
   - the number of final candidates after progressive filtering.

5. **`initial_score_size`**: `int`
   - the size of the score set used to score the examples, a larger number usually means a better filtering but may cause the process to slow down.

6. **`iter_num`**: `int`
   - the number of iterations diversity guided search takes.
 
7. **`beam_size`**: `int`
   - beam size B, larger B means more permutations are considered.

8. **`substitution_size`**: `int`
   - the number of permutations within each beam that are to be substituted, rest will be shuffled.

9. **`alpha`**: `float`
   - trade-off between more information score and less similar, a higher value means similarity is regarded more important, see original paper equation (5) for more detail.

##### Note：At the moment information scores are calculated by comparing the outputs of the LLM rather than using the probabilities of the outcomes as the original method indicated.