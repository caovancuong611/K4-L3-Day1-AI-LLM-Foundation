# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, các lần gọi lặp lại gần như cho ra đúng một câu trả lời — rất ổn định, ít đa dạng từ vựng. Khi tăng lên 0.5 rồi 1.0, cách diễn đạt bắt đầu thay đổi và model có xu hướng chọn những sự thật khác nhau giữa các lần gọi, nhưng vẫn giữ ngữ pháp mạch lạc. Ở 1.5, phản hồi kém ổn định hơn hẳn: câu văn đôi khi lủng củng hoặc pha trộn thông tin thiếu chính xác — cho thấy temperature càng cao thì độ "sáng tạo" tăng nhưng độ tin cậy giảm.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần câu trả lời nhất quán, chính xác và có thể dự đoán được giữa các lần hỏi giống nhau — khách hàng không nên nhận hai câu trả lời khác hẳn nhau cho cùng một câu hỏi về chính sách hay hướng dẫn sử dụng, vì sự thiếu nhất quán đó làm giảm độ tin cậy của thương hiệu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Giá output của GPT-4o (0.010 USD/1K token) so với GPT-4o-mini (0.0006 USD/1K token) chênh nhau xấp xỉ **16.7 lần**. Với workload này (10.000 người dùng × 3 lượt × 350 token ≈ 10.500.000 token output/ngày), GPT-4o tốn ước tính ~105 USD/ngày trong khi mini chỉ ~6.3 USD/ngày. GPT-4o xứng đáng dùng cho các tác vụ cần suy luận sâu, độ chính xác cao (tư vấn pháp lý, phân tích hợp đồng, debug code phức tạp); GPT-4o-mini phù hợp cho câu hỏi thường gặp (FAQ), phân loại yêu cầu hoặc trả lời ngắn — nơi tốc độ và chi phí quan trọng hơn độ sâu suy luận.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona "giáo viên tiểu học", phản hồi thường ngắn hơn, dùng từ vựng đơn giản và ví dụ gần gũi đời sống (ví dụ so sánh blockchain với một cuốn sổ ghi chép chung mà cả lớp cùng xem được), tránh gần như hoàn toàn thuật ngữ chuyên môn. Với persona "chuyên gia tài chính", phản hồi dài và chi tiết hơn, dùng thuật ngữ kỹ thuật (sổ cái phân tán, cơ chế đồng thuận, hàm băm, hợp đồng thông minh...) và đi sâu vào cơ chế vận hành hoặc ứng dụng thực tế. Cùng một câu hỏi nhưng system prompt định hình gần như hoàn toàn giọng điệu, độ sâu và đối tượng giả định của câu trả lời — cho thấy persona là công cụ mạnh để tùy biến hành vi model mà không cần đổi model hay fine-tune.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt ~100 từ, số token đếm bằng `count_tokens` (tiktoken) thường cao hơn ước lượng `số từ / 0.75` khoảng 40–70% (ví dụ ước lượng ~133 token nhưng tiktoken thường cho ra ~190–220 token). Chênh lệch này xảy ra vì bộ mã hóa (BPE) được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên các ký tự Unicode có dấu của tiếng Việt (ă, â, ê, ô, ơ, ư và các dấu thanh) hiếm khi nằm trọn trong một token phổ biến — một từ tiếng Việt có dấu thường bị tách thành 2–3 token nhỏ thay vì 1 token như một từ tiếng Anh tương đương.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi câu trả lời dài và người dùng đang chờ trong một giao diện tương tác thời gian thực, như chatbot hay trợ lý lập trình — thấy chữ xuất hiện ngay giúp cảm giác phản hồi nhanh hơn hẳn, giảm cảm giác chờ đợi dù tổng thời gian model xử lý không đổi. Non-streaming lại phù hợp hơn khi ứng dụng cần toàn bộ phản hồi trước khi dùng, ví dụ parse kết quả thành JSON, kiểm duyệt nội dung trước khi hiển thị, hay chạy các tác vụ batch không có người dùng theo dõi trực tiếp — lúc đó việc xử lý từng chunk chỉ thêm độ phức tạp mà không mang lại lợi ích trải nghiệm nào.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn thời gian chờ ra xa dần sau mỗi lần thất bại, nên áp lực lên server giảm dần theo thời gian thay vì giữ nguyên một mức áp lực cố định suốt quá trình retry. Nếu hàng nghìn client cùng dùng delay cố định giống nhau (luôn chờ đúng 1 giây), chúng sẽ đồng loạt retry cùng lúc thành từng đợt — mỗi đợt lại tạo ra một cơn dồn dập request mới ("thundering herd"), khiến server vừa hồi phục một chút đã bị quá tải trở lại ngay lập tức, có thể kéo dài sự cố thay vì giải quyết nó.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Từ "thân thiện" định hướng giọng điệu gần gũi, không formal cứng nhắc, phù hợp với học viên mới bắt đầu. Yêu cầu "trả lời ngắn gọn" quan trọng vì `max_tokens` và history nhiều lượt có giới hạn — câu trả lời dài dễ vượt giới hạn token và làm history phình to nhanh, tốn thêm chi phí ở mỗi lượt sau. Chỉ định rõ "bằng tiếng Việt" đảm bảo model không tự chuyển sang tiếng Anh khi gặp thuật ngữ kỹ thuật, giữ trải nghiệm nhất quán cho người học Việt Nam.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất, nên trợ lý "quên" hoàn toàn những gì đã nói trước đó — nếu người dùng tham chiếu lại một chi tiết từ 4–5 lượt trước, model không còn ngữ cảnh để trả lời chính xác. Cải thiện đề xuất: thêm một bước tóm tắt (summarization) — mỗi khi history sắp bị cắt bớt (`history[-6:]`), gọi thêm một lời gọi API nhỏ để tóm tắt các message sắp bị loại thành 1–2 câu ngắn, lưu vào một biến `long_term_summary`, rồi chèn biến này vào đầu system prompt ở các lượt sau. Cách này giữ lại ý chính của cuộc hội thoại cũ mà không tốn nhiều token như giữ nguyên toàn bộ lịch sử.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
