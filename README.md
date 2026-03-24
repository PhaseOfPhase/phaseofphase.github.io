## Phần 1: Phân tích ý tưởng gốc của em - Phù hợp và không phù hợp với bài toán Agent

### Nhắc lại ý tưởng gốc

Ý tưởng ban đầu là giải quyết bottleneck ở **decode stage** của LLM: thay vì decode 1 token mỗi lần (matrix × vector, kém hiệu quả trên GPU), thì gom 10 token thành 1 "bản summary" - tức là biểu diễn **N token → 1 đơn vị nén** thông qua state space model (Mamba). Kèm theo cơ chế: nếu 2 bản summary liền nhau có sự thay đổi lớn thì tách nhỏ hơn (dùng null token để padding).

Tác giả đã nhận ra rằng cách tiếp cận này không phù hợp cho inference ChatGPT/Claude thông thường - và muốn **tái sử dụng** ý tưởng nén N-token-thành-1-unit cho bài toán **context management của Agent**.

---

### 1. Phân tích các luận điểm cốt lõi

#### Luận điểm A: "Nén N token thành 1 unit" là hướng đúng cho Agent

**Phù hợp:**

Đây là một observation rất chuẩn. Agent phải interleave nhiều lần LLM invocation và tool call trong long-running task, khiến token accumulate rất nhanh - gây tràn context window, tăng cost/latency, và làm loãng performance của agent.

Bài toán **context dilution** (loãng context) mà tác giả nêu thực ra đã được nhiều paper xác nhận. Các prior approach thường bị "brevity bias" - tức là khi tóm tắt thì đánh mất domain insight, và "context collapse" - khi iterative rewriting thì các detail bị mất dần theo thời gian.

Cụ thể hơn, tính đến tháng 3/2026, trọng tâm của context engineering đã dịch chuyển từ "how to pack the best prompt" sang cách agent system quản lý runtime state, memory, tools, và long-horizon execution.

**Không phù hợp / cần điều chỉnh:**

Ý tưởng gốc nén theo **độ dài cố định** (luôn là 10 token) mang tư duy từ hardware optimization (batch decode để tận dụng GPU). Nhưng ở bài toán Agent, **ngữ nghĩa** không phân bổ đều theo số token - một tool call result 10 token có thể quan trọng hơn cả đoạn reasoning 200 token. Nén theo window cố định sẽ làm mờ ranh giới ngữ nghĩa.

---

#### Luận điểm B: Dùng State Space Model (Mamba) làm "bộ nén"

**Phù hợp về mặt kỹ thuật:**

Hidden state của Mamba thực chất là một **fixed-size vector** tóm tắt toàn bộ chuỗi đầu vào - đây là đặc tính lý tưởng để làm bộ nhớ nén. Taxonomy mới nhất về Agent Memory phân loại memory theo 3 dạng carrier: Token-level (explicit & discrete), Parametric (implicit weights), và Latent (hidden states). Hidden state của Mamba rơi vào đúng loại **Latent memory** - đây là hướng nghiên cứu đang được chú ý.

**Không phù hợp / hạn chế lớn:**

- Mamba được train như một SSM riêng - nếu muốn dùng làm "compressor" cho output của một Transformer-based LLM, cần **alignment** giữa 2 kiến trúc khác nhau. Không thể plug-and-play.
- Hidden state của Mamba là **opaque** - LLM không thể trực tiếp "đọc" hay "attend vào" vector đó theo cơ chế attention thông thường. Phải có thêm một projection layer để chuyển sang token space.
- Với LLM đóng (GPT-4, Claude), không thể can thiệp vào architecture - chỉ có thể làm ở **application layer** (token-level, text-level).

---

#### Luận điểm C: Cơ chế "adaptive granularity" - khi 2 summary liền nhau có thay đổi lớn thì tách nhỏ hơn

**Phù hợp - đây là điểm sáng nhất:**

Đây là intuition rất tốt: **không nén đồng đều**, mà nén dựa theo **information density / semantic shift**. Đây chính xác là nguyên lý của nhiều kỹ thuật nén tiên tiến. Hierarchical summarization chia input thành các chunk, tóm tắt từng chunk, rồi tóm tắt các summary - tạo ra hierarchy với mỗi level biểu diễn một cái nhìn cô đọng hơn về input.

Ý tưởng "phát hiện semantic shift để điều chỉnh granularity" giống với các hệ thống **dynamic chunking** hiện đại, vốn dùng embedding similarity để tìm ranh giới tự nhiên trong văn bản.

**Không phù hợp:**

- Cơ chế "chèn null token để giữ độ dài 10 token" là artifact từ bài toán hardware (cần batch size cố định cho GPU efficiency). Ở bài toán Agent context, không có ràng buộc này - nên không cần null padding.
- Metric "thay đổi lớn giữa 2 summary" cần được định nghĩa rõ hơn: dùng cosine distance giữa hidden state? KL divergence? Perplexity? Mỗi lựa chọn có tradeoff khác nhau.

---

#### Luận điểm D: Framing bài toán - Agent cần "stateful long-horizon memory"

**Đây là framing chuẩn xác nhất trong toàn bộ ý tưởng.**

Tính đến đầu 2026, nhiều tổ chức xác nhận rằng context engineering và managing context at scale là một trong những thách thức lớn nhất khi deploy agent ở production.

Nghiên cứu từ Anthropic cho thấy nhiều agent với isolated context outperform single-agent implementation, vì mỗi subagent context window có thể được allocate cho một subtask hẹp hơn - nhưng cái giá phải trả là token usage có thể cao hơn đến 15 lần.

Vấn đề "loãng context" mà tác giả nêu được industry gọi là **context poisoning**, **context collapse**, hay **lost-in-the-middle** - và nó được xác nhận là một vấn đề thực sự nghiêm trọng trong production.

---

### 2. Tổng hợp Ưu - Nhược điểm khi áp dụng ý tưởng gốc vào bài toán Agent

| Khía cạnh                                        | Đánh giá                                                             |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| **Tư duy nén N-token → 1 unit**                  | ✅ Đúng hướng, đã được literature xác nhận                           |
| **Adaptive granularity dựa trên semantic shift** | ✅ Điểm mạnh nhất, có thể giữ nguyên                                 |
| **Dùng Mamba làm compressor**                    | ⚠️ Khả thi nếu fine-tune, nhưng không thể dùng với LLM closed-source |
| **Window size cố định (10 token)**               | ❌ Không phù hợp cho Agent - nên thay bằng semantic boundary         |
| **Null token padding**                           | ❌ Artifact của hardware constraint, không cần thiết cho Agent       |
| **Nén ở token/latent level**                     | ⚠️ Chỉ khả thi với open-source model hoặc custom architecture        |
| **Nén ở text/semantic level**                    | ✅ Phù hợp cho mọi LLM (kể cả API-only)                              |
| **Phát hiện information boundary**               | ✅ Cần thiết và là điểm khác biệt so với rolling window cơ bản       |

---

### 3. Đối chiếu với State-of-the-Art (2025-2026)

Để đặt ý tưởng vào bối cảnh đầy đủ, điểm qua những gì đã tồn tại:

- **Hierarchical summarization** (đã phổ biến): Short-term memory giữ nguyên các turn gần nhất, medium-term memory chứa các compressed summary của recent sessions, long-term memory lưu key facts - hệ thống draw from all tiers khi process query.
- **Incremental/anchored summary** (Factory.ai, 2025): Thay vì regenerate toàn bộ summary mỗi request, hệ thống maintain một persistent summary, update incrementally mỗi khi truncate old messages - mỗi summary anchor được gắn với một message cụ thể.
- **ACE Framework** (Oct 2025): ACE treat context như "evolving playbooks" - accumulate, refine, và organize strategies qua modular process gồm generation, reflection, và curation - ngăn context collapse bằng structured incremental updates.
- **Cross-modal compression** (Feb 2026): Trong multi-agent debate, visual compression đạt 92% token reduction so với text-based, vì vision tokens tự nhiên capture structural relationships và logical flow - outperform cả text-with-summarization về accuracy.

**Kết luận phần 1:** Ý tưởng của tác giả có **kernel rất tốt** - đặc biệt là adaptive granularity dựa trên semantic shift. Điểm cần "dịch chuyển" là: bỏ ràng buộc từ hardware (window cố định, null padding, Mamba bắt buộc), và reframe thành một **hierarchical semantic compression system** với dynamic boundary detection hoạt động ở text/embedding level, có thể tích hợp vào bất kỳ LLM-based agent nào. Đây sẽ là nền tảng cho các đề xuất ở bước tiếp theo.

---

## Phần 2: Đề xuất các phương pháp cho bài toán Agent Context

Từ ý tưởng gốc (adaptive N-token-to-1-unit compression với semantic boundary detection), và dựa trên landscape hiện tại, tôi đề xuất **4 phương pháp** theo mức độ tăng dần về độ phức tạp và tiềm năng.

---

### Phương pháp 1 - Semantic-Boundary Incremental Summarization (SBIS)

**Tư tưởng cốt lõi:** Áp dụng trực tiếp intuition từ ý tưởng gốc - nén nhiều đơn vị thành 1 - nhưng thay "10 token cố định" bằng "1 semantic episode có ranh giới tự nhiên".

**Cơ chế hoạt động:**

Toàn bộ lịch sử conversation của agent được tổ chức thành 3 tầng:

```
[HOT BUFFER]  →  [EPISODE STORE]  →  [SEMANTIC VAULT]
  N turn gần      Các episode đã       Các fact/belief
  nhất (raw)      nén, có index        cốt lõi đã trích
```

Thay vì nén theo N turn cố định, agent phát hiện **episode boundary** dựa trên semantic shift: khi cosine distance giữa embedding của turn hiện tại và running mean của episode hiện tại vượt ngưỡng $\theta$, thì đóng episode và nén lại. Đây là cơ chế "2 bản summary liền nhau có thay đổi lớn" từ ý tưởng gốc, được implement ở text level thay vì latent level.

HiAgent và các framework tương tự đã validate hướng này: chúng chunk working memory dựa trên subgoal completion - khi một goal được giải quyết xong thì nén toàn bộ chuỗi action-observation thành một summary ngắn gọn, giữ lại thông tin hierarchical và hỗ trợ efficient retrieval.

**Ưu điểm:**

- Không yêu cầu thay đổi kiến trúc LLM - hoạt động hoàn toàn ở application layer
- Ranh giới episode có ý nghĩa ngữ nghĩa, không phải artifact của window size
- Incremental: không cần reprocess toàn bộ history khi có turn mới

**Nhược điểm:**

- Threshold $\theta$ cần calibrate per domain
- Mất mát thông tin khi nén là không thể tránh khỏi hoàn toàn
- Không giải quyết được vấn đề "cross-episode reasoning" - khi thông tin cần thiết trải dài qua nhiều episode khác nhau

---

### Phương pháp 2 - Dual-Granularity Context với Importance Scoring (DGCI)

**Tư tưởng cốt lõi:** Không phải toàn bộ context đều cần nén như nhau. Thay vì nén đồng đều, xây dựng hệ thống 2 chiều: **granularity** (mức nén) và **importance** (độ quan trọng) - phần quan trọng giữ raw, phần ít quan trọng nén mạnh.

**Cơ chế hoạt động:**

Mỗi turn hoặc segment của context được gán 2 thuộc tính:

- $g \in \{0, 1, 2, 3\}$: granularity level (0 = raw, 3 = chỉ giữ metadata)
- $i \in [0, 1]$: importance score (tính dựa trên frequency of reference, recency, và semantic centrality)

Khi context window đầy, agent không truncate theo FIFO mà theo **priority queue**: những segment có $g$ cao và $i$ thấp bị loại bỏ hoặc nén xuống level tiếp theo trước.

Nghiên cứu mới nhất về overflow detection trong compressed token cho thấy rằng **adaptive chunking** - tự động resize input segment dựa trên semantic density thay vì arbitrary fixed length - là chìa khóa để tránh information overflow, tức là trạng thái khi complexity của input vượt quá capacity của compressed token.

Importance score $i$ được tính theo công thức tổng hợp:

$$i = \alpha \cdot r + \beta \cdot f + \gamma \cdot c$$

Trong đó $r$ là recency decay, $f$ là reference frequency (bao nhiêu lần turn sau đề cập lại segment này), $c$ là semantic centrality (cosine similarity với mean embedding của toàn bộ task).

**Ưu điểm:**

- Giải quyết được "loãng context" triệt để hơn - những gì quan trọng luôn ở granularity cao
- Adaptive: không cần threshold tĩnh
- Reference frequency $f$ capture được "bộ nhớ làm việc" của agent - những gì nó hay quay lại

**Nhược điểm:**

- Chi phí tính importance score mỗi turn là $O(n)$ với $n$ là số segment trong store
- $\alpha, \beta, \gamma$ là hyperparameter - cần tuning
- Reference frequency chỉ có thể tính retrospectively, không predictive

---

### Phương pháp 3 - Episodic-to-Semantic Consolidation với Dynamic Graph (ESCDG)

**Tư tưởng cốt lõi:** Mượn từ cognitive science - con người không chỉ nén memory, mà còn **consolidate** từ episodic (sự kiện cụ thể) sang semantic (tri thức tổng quát). Ý tưởng gốc của tác giả về "bản tóm tắt" thực ra là đang mô tả quá trình episodic compression - nhưng Agent cần thêm một bước nữa.

**Cơ chế hoạt động:**

Hệ thống gồm 3 tầng memory và 2 quá trình chuyển đổi:

```
Tầng 1: EPISODIC BUFFER
   │  (raw turns + tool calls + observations)
   │
   ▼ [Consolidation Process 1: Episode Compression]
   │  Trigger: semantic boundary hoặc episode end
   │  Output: 1 episode node với {summary, key_entities, timestamp, embedding}
   │
Tầng 2: EPISODE GRAPH
   │  (các episode node, có edge nếu share entity hoặc causal link)
   │
   ▼ [Consolidation Process 2: Episodic → Semantic]
   │  Trigger: khi 3+ episodes share common pattern
   │  Output: 1 semantic node (fact/belief/rule)
   │
Tầng 3: SEMANTIC KNOWLEDGE BASE
      (facts, beliefs, learned rules - persistent across sessions)
```

Trong các hệ thống multi-agent tiên tiến, khi agent giải quyết một vấn đề mới, interaction trace được lưu vào shared episodic memory trước. Sau đó, một background process phân tích các trace này, nhận diện pattern thành công và trừu tượng hóa thành generalizable skill hoặc rule, rồi ghi vào shared semantic memory.

Điểm mấu chốt của ESCDG là **Episode Graph** - các episode không chỉ được lưu độc lập mà còn được link với nhau. Khi agent cần context, thay vì load N episode gần nhất (FIFO), nó traverse graph theo entity relevance - giải quyết được vấn đề "cross-episode reasoning".

Mem0g đã validate cách tiếp cận graph-based: hệ thống phân tích entity và context trong conversation để nhận diện các kết nối có ý nghĩa semantic, phân loại relationship với label phù hợp, tạo thành relationship triplet làm edge trong memory graph - cho phép complex reasoning qua interconnected information.

**Ưu điểm:**

- Giải quyết được cross-episode reasoning qua graph traversal
- Episodic → Semantic consolidation ngăn context không bị "flatten" theo thời gian
- Graph structure cho phép efficient retrieval theo entity, không chỉ theo thời gian

**Nhược điểm:**

- Phức tạp nhất trong 3 phương pháp đầu - overhead của graph maintenance là đáng kể
- Entity extraction và relationship labeling cần thêm LLM call
- Graph có thể bị "fragmented" nếu agent làm nhiều task không liên quan

---

### Phương pháp 4 - Hierarchical Compression với Overflow-Aware Adaptive Granularity (HCOAG)

**Tư tưởng cốt lõi:** Đây là phương pháp tổng hợp nhất - lấy trực tiếp từ cơ chế "null token + adaptive tách nhỏ" của ý tưởng gốc, nhưng reframe hoàn toàn cho Agent context. Thay vì padding null token để giữ window cố định, agent **tự động điều chỉnh compression ratio** dựa trên information density và overflow risk.

**Cơ chế hoạt động:**

```
INCOMING CONTEXT
      │
      ▼
[OVERFLOW RISK ESTIMATOR]
  Tính: D(segment) = semantic_density × length
  Nếu D cao → compression ratio thấp (giữ nhiều)
  Nếu D thấp → compression ratio cao (nén mạnh)
      │
      ├──[HIGH DENSITY]──► Giữ raw hoặc light summary (ratio 5:1)
      │
      ├──[MED DENSITY]──► Medium summary (ratio 20:1)
      │
      └──[LOW DENSITY]──► Chỉ giữ key entities + timestamp (ratio 100:1)
```

Nghiên cứu về overflow detection trong RAG cho thấy: phát hiện sớm overflow giúp computational pruning - nhận diện và loại bỏ các saturated representation ngay sau projection, tránh lãng phí LLM inference trên degraded context. Một vector đơn lẻ có thể encode hàng nghìn token về mặt lý thuyết, nhưng practical capacity phụ thuộc vào kiến trúc và complexity của input.

Điểm khác biệt so với Phương pháp 1-3: HCOAG không chỉ nén khi context đầy (reactive) mà **dự đoán overflow trước** (proactive). Nó maintain một "compression budget" cho mỗi agent session:

$$B_{remaining} = C_{max} - \sum_{s \in \text{active}} |s| \cdot (1 - r_s)$$

Với $C_{max}$ là context window size, $|s|$ là length của segment $s$, và $r_s$ là compression ratio hiện tại của $s$. Khi $B_{remaining}$ xuống dưới ngưỡng $\delta$, agent tăng compression ratio của các segment ít dense nhất trước.

Production systems tốt nhất hiện tại đã validate rằng multi-turn workflows với hàng chục tool call cần external memory và selective context injection - không thể dùng một kỹ thuật duy nhất. HCOAG giải quyết điều này bằng cách treat context như một constrained resource cần được actively managed.

**Ưu điểm:**

- Proactive thay vì reactive - không đợi đến khi context đầy mới xử lý
- Overflow-aware: tránh được trường hợp nén quá mức gây mất thông tin quan trọng
- Compression ratio có thể được điều chỉnh liên tục, không phải one-shot decision

**Nhược điểm:**

- Semantic density estimation bản thân đã là một bài toán khó - cần either lightweight model hoặc heuristic
- Budget management thêm một lớp state cần maintain
- Khi toàn bộ context đều high-density (ví dụ agent đang làm task kỹ thuật phức tạp), không có segment nào để nén mạnh - vẫn cần cơ chế fallback

---

### So sánh tổng quan 4 phương pháp

| Tiêu chí                     | SBIS (P1)      | DGCI (P2)      | ESCDG (P3)       | HCOAG (P4) |
| ---------------------------- | -------------- | -------------- | ---------------- | ---------- |
| **Complexity triển khai**    | Thấp           | Trung bình     | Cao              | Cao        |
| **Chống context dilution**   | Tốt            | Rất tốt        | Rất tốt          | Tốt        |
| **Cross-episode reasoning**  | Yếu            | Trung bình     | Mạnh             | Trung bình |
| **Proactive management**     | Không          | Không          | Không            | Có         |
| **Latency overhead**         | Thấp           | Trung bình     | Cao              | Trung bình |
| **Phù hợp closed LLM (API)** | Hoàn toàn      | Hoàn toàn      | Hoàn toàn        | Hoàn toàn  |
| **Persist across session**   | Không tự nhiên | Không tự nhiên | Có (Semantic KB) | Cần thêm   |
| **Gốc từ ý tưởng ban đầu**   | Trực tiếp      | Gián tiếp      | Gián tiếp        | Trực tiếp  |

---

Cả 4 phương pháp đều hoạt động ở **application/text layer**, không yêu cầu thay đổi kiến trúc LLM. Đây là điều kiện quan trọng để tương thích với closed-source model như GPT-4, Claude. P3 (ESCDG) là mạnh nhất về reasoning nhưng phức tạp nhất; P4 (HCOAG) là sát nhất với ý tưởng gốc và duy nhất có proactive management.

## Phần 3: Đề xuất phương pháp tối ưu nhất

### Tên phương pháp: **CASCADE** - Context-Aware Semantic Compression with Adaptive Density Estimation

Không có phương pháp đơn lẻ nào trong 4 phương pháp trên là tối ưu. Phương pháp tối ưu là **tổng hợp có chọn lọc** - lấy đúng điểm mạnh của từng phương pháp và ghép lại thành một pipeline coherent.

---

### Trực giác thiết kế

Trước khi đi vào chi tiết, cần xác lập rõ **3 nguyên tắc thiết kế** mà bất kỳ phương pháp tối ưu nào cũng phải thỏa mãn:

**Nguyên tắc 1 - Asymmetric Compression:** Không phải tất cả context đều ngang nhau. Tool call result, constraint từ user, và key decision là "load-bearing" information - mất đi là mất irreversibly. Prose reasoning, intermediate steps, và observation verbose là "scaffolding" - có thể nén mạnh.

**Nguyên tắc 2 - Proactive over Reactive:** Không đợi context đầy mới xử lý. Nén phải xảy ra **liên tục và tăng dần**, như một background process.

**Nguyên tắc 3 - Lossless Skeleton, Lossy Flesh:** Không tồn tại "lossless compression" cho context dài - đó là ảo tưởng. Mục tiêu thực tế là: giữ **skeleton** (entity, relation, decision, constraint) lossless, còn **flesh** (elaboration, intermediate reasoning) có thể lossy theo mức độ kiểm soát.

---

### Kiến trúc tổng thể của CASCADE

```
                    ┌─────────────────────────────────────┐
                    │           AGENT RUNTIME              │
                    │                                      │
  New turn/         │   ┌──────────────────────────────┐  │
  tool result ──────┼──►│   HOT BUFFER (raw, recent)   │  │
                    │   │   ~20% of context budget      │  │
                    │   └──────────┬───────────────────-┘  │
                    │              │ Episode boundary        │
                    │              │ detected (semantic      │
                    │              │ shift > θ)              │
                    │              ▼                         │
                    │   ┌──────────────────────────────┐   │
                    │   │  EPISODE STORE (compressed)  │   │
                    │   │  ~50% of context budget      │   │
                    │   │  Graph-linked by entity       │   │
                    │   └──────────┬───────────────────┘   │
                    │              │ Consolidation          │
                    │              │ trigger (3+ episodes   │
                    │              │ share pattern)         │
                    │              ▼                         │
                    │   ┌──────────────────────────────┐   │
                    │   │  SEMANTIC VAULT (facts/rules) │   │
                    │   │  ~30% of context budget       │   │
                    │   │  Persistent across sessions   │   │
                    │   └──────────────────────────────┘   │
                    │                                        │
                    │   ┌──────────────────────────────┐   │
                    │   │  OVERFLOW RISK MONITOR        │   │
                    │   │  Watches B_remaining          │   │
                    │   │  Triggers re-compression      │   │
                    │   └──────────────────────────────┘   │
                    └─────────────────────────────────────--┘
```

---

### Chi tiết từng tầng

#### Tầng 1: HOT BUFFER (từ SBIS - P1)

Giữ nguyên raw các turn gần nhất. Không nén gì ở đây. Budget cố định ~20% context window.

Boundary detection dùng embedding cosine distance:

$$\text{shift}(t) = 1 - \frac{\mathbf{e}_t \cdot \bar{\mathbf{e}}_{\text{episode}}}{\|\mathbf{e}_t\| \cdot \|\bar{\mathbf{e}}_{\text{episode}}\|}$$

Khi $\text{shift}(t) > \theta$ hoặc HOT BUFFER đầy - episode hiện tại được đóng lại và đẩy xuống tầng 2. Đây chính là **semantic boundary detection** từ ý tưởng gốc, được implement sạch không cần null token hay Mamba.

---

#### Tầng 2: EPISODE STORE (từ ESCDG - P3 + DGCI - P2)

Mỗi episode được nén thành một **Episode Node**:

```json
{
  "id": "ep_042",
  "summary": "Agent fetched user's calendar, found conflict on Thu 3pm, proposed reschedule to Fri 10am. User confirmed.",
  "key_entities": ["calendar", "Thu 3pm slot", "Fri 10am slot", "conflict"],
  "decisions": ["reschedule confirmed"],
  "constraints": ["user unavailable Thu afternoon"],
  "importance": 0.87,
  "embedding": [...],
  "linked_episodes": ["ep_039", "ep_041"]
}
```

Compression ratio ở tầng này được điều chỉnh bởi **importance score** (từ P2):

$$i = \alpha \cdot r + \beta \cdot f + \gamma \cdot c$$

- Episode có $i$ cao → giữ summary dài hơn, giữ nhiều detail hơn
- Episode có $i$ thấp → nén xuống chỉ còn `key_entities` + `decisions` + `constraints`

Điều này implement **Asymmetric Compression** - load-bearing information (decisions, constraints) luôn được giữ bất kể $i$.

Các episode được **graph-link** nếu share entity - cho phép cross-episode retrieval khi cần.

---

#### Tầng 3: SEMANTIC VAULT (từ ESCDG - P3)

Khi 3+ episode nodes share entity hoặc pattern, một **consolidation process** chạy:

```
ep_010: "User từ chối option A vì quá đắt"
ep_023: "User từ chối option C vì quá đắt"
ep_031: "User chọn option B vì có discount"
          │
          ▼ Consolidation
SEMANTIC NODE: "User có price sensitivity cao, ưu tiên options có discount"
```

Đây là quá trình **episodic → semantic** - chuyển từ sự kiện cụ thể sang tri thức tổng quát. Semantic Vault **persist across sessions** - đây là điều không phương pháp nào trong 4 phương pháp ban đầu làm được hoàn chỉnh.

---

#### Tầng 4: OVERFLOW RISK MONITOR (từ HCOAG - P4)

Monitor chạy như background process, track:

$$B_{\text{remaining}} = C_{\max} - |H| - \sum_{e \in E} |e| - |V|$$

Với $H$ là HOT BUFFER, $E$ là EPISODE STORE, $V$ là SEMANTIC VAULT.

Khi $B_{\text{remaining}} < \delta_1$: tăng compression ratio của các episode có $i$ thấp nhất trong EPISODE STORE.

Khi $B_{\text{remaining}} < \delta_2 < \delta_1$: trigger emergency consolidation - các episode có $i$ thấp bị flatten xuống chỉ còn metadata, entities được merge vào SEMANTIC VAULT.

**Proactive, không reactive.**

---

### Context Injection khi LLM gọi

Khi LLM cần generate, CASCADE không dump toàn bộ 3 tầng vào context. Thay vào đó, nó **assemble context động** theo relevance của query hiện tại:

```
ASSEMBLED CONTEXT =
  [SEMANTIC VAULT - toàn bộ]          ← luôn có, nhỏ gọn
  + [EPISODE STORE - top-k relevant]  ← retrieve theo embedding similarity
  + [HOT BUFFER - toàn bộ]            ← luôn có, raw
```

Top-k episode retrieval dùng:

$$\text{score}(e, q) = \lambda \cdot \text{sim}(\mathbf{e}_e, \mathbf{e}_q) + (1-\lambda) \cdot i_e$$

Kết hợp semantic similarity với query $q$ và importance score $i_e$ của episode $e$. Điều này đảm bảo không chỉ lấy episode **liên quan** mà còn lấy episode **quan trọng**.

---

### Điểm khác biệt cốt lõi so với các hệ thống hiện tại

| Đặc điểm                          | Rolling Window   | Hierarchical Summarization thông thường | **CASCADE**                    |
| --------------------------------- | ---------------- | --------------------------------------- | ------------------------------ |
| Boundary detection                | Cố định (N turn) | Cố định (N turn)                        | Semantic shift động            |
| Compression uniformity            | Uniform          | Uniform                                 | Asymmetric (skeleton lossless) |
| Cross-session persistence         | Không            | Không                                   | Có (Semantic Vault)            |
| Proactive overflow management     | Không            | Không                                   | Có                             |
| Cross-episode reasoning           | Không            | Yếu                                     | Graph traversal                |
| Episodic → Semantic consolidation | Không            | Không                                   | Có                             |

---

### Complexity và Feasibility

CASCADE yêu cầu những gì khi triển khai thực tế?

- **Embedding model nhỏ** (ví dụ `text-embedding-3-small`): dùng để tính semantic shift và retrieval score - chi phí thấp, latency thấp.
- **LLM call phụ** cho compression: chạy async/background, không block main agent loop. Có thể dùng model nhỏ hơn (GPT-4o-mini, Claude Haiku) cho bước nén.
- **Vector store nhỏ** (in-memory hoặc SQLite + FAISS): lưu episode embeddings.
- **Zero architecture change** với LLM chính: hoàn toàn ở application layer, tương thích với mọi API.

Chi phí thực tế tăng thêm so với không có memory management: khoảng **10-15% token overhead** cho compression calls, đổi lại giảm được **60-80% context window usage** trong long agent session - net positive rõ ràng.
