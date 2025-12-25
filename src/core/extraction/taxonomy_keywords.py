"""
Comprehensive Event Keyword Taxonomy for Vietnamese.
Categorized into 7 distinct usage-based groups as per proposal.md.
"""

EVENT_KEYWORDS_VI = {
    "T1_CRISIS": {
        "Accidents": [
            "tai nạn", "va chạm", "đâm", "tông",
            "lật xe", "cháy", "nổ", "phát nổ",
            "sập", "sập cầu", "sập nhà",
            "rơi", "trật bánh", "trật ray",
            "máy bay rơi", "tàu hỏa trật bánh",
            "bị thương", "nhập viện",
            "tử vong", "thiệt mạng", "chết người"
        ],
        "Disasters": [
            "cơn bão", "áp thấp", "lốc xoáy",
            "mưa lớn", "mưa to", "ngập", "ngập lụt",
            "lũ", "lũ quét", "sạt lở", "sạt lở đất",
            "động đất", "dư chấn",
            "sóng thần", "núi lửa"
        ],
        "Epidemics": [
            "dịch bệnh", "bùng phát",
            "virus", "vi rút",
            "lây nhiễm", "ca nhiễm",
            "dịch", "ổ dịch",
            "covid", "cúm", "cúm gia cầm",
            "sốt xuất huyết", "tay chân miệng",
            "cách ly", "phong tỏa"
        ],
        "Crimes": [
            "giết người", "án mạng",
            "bắn", "nổ súng",
            "cướp", "cướp giật",
            "bắt cóc", "xâm hại", "hiếp dâm",
            "khủng bộ", "đánh bom",
            "bị bắt", "bắt giữ",
            "khởi tố", "điều tra"
        ]
    },

    "T2_GOVERNANCE": {
        "Policy": [
            "nghị định", "thông tư", "luật", "dự thảo",
            "ban hành", "quy định mới", "điều chỉnh",
            "thuế", "lệ phí", "thủ tục hành chính",
            "cải cách", "đề xuất", "kiến nghị"
        ],
        "Statements": [
            "phát biểu", "chỉ đạo", "yêu cầu",
            "thủ tướng", "bộ trưởng", "chính phủ",
            "quốc hội", "ubnd", "họp báo"
        ]
    },

    "T3_REPUTATION": {
        "Controversy": [
            "tranh cãi", "lùm xùm",
            "scandal", "phốt",
            "bị tố", "tố cáo",
            "bị chỉ trích", "bị phản đối",
            "làn sóng phản đối",
            "tẩy chay", "xin lỗi công khai",
            "từ chức", "bị đình chỉ"
        ]
    },

    "T4_MARKET": {
        "Economy": [
            "lạm phát", "giảm phát",
            "suy thoái", "tăng trưởng",
            "kinh tế chậm lại",
            "lãi suất", "tăng lãi suất", "giảm lãi suất",
            "thị trường chứng khoán",
            "cổ phiếu", "trái phiếu",
            "tỷ giá", "tiền tệ",
            "phá sản", "cắt giảm nhân sự",
            "sa thải"
        ],
        "Tech": [
            "trí tuệ nhân tạo", "AI",
            "bị hack", "tấn công mạng",
            "rò rỉ dữ liệu",
            "sập hệ thống", "lỗi hệ thống",
            "gián đoạn dịch vụ",
            "ra mắt sản phẩm",
            "bản cập nhật"
        ],
        "Consumer": [
            "xu hướng tiêu dùng", "mua sắm",
            "giảm giá", "khuyến mãi",
            "shopee", "tiktok shop", "lazada",
            "lifestyle", "phong cách sống"
        ]
    },

    "T5_CULTURE": {
        "Viral": [
            "viral", "lan truyền",
            "gây bão mạng", "bão mạng", "gây sốt",
            "xu hướng", "trending",
            "được chia sẻ", "triệu lượt xem",
            "clip nóng", "video gây sốt"
        ],
        "Entertainment": [
            "phim mới", "ra rạp",
            "doanh thu phòng vé",
            "concert", "buổi hòa nhạc",
            "lễ hội âm nhạc",
            "nghệ sĩ", "ca sĩ", "diễn viên",
            "hẹn hò", "kết hôn", "ly hôn",
            "tin đồn tình cảm"
        ],
        "Celebs": [
            "người nổi tiếng", "idol", "thần tượng",
            "showbiz", "ngôi sao", "người mẫu"
        ]
    },

    "T6_OPERATIONAL": {
        "PublicServices": [
            "ùn tắc", "kẹt xe",
            "mất điện", "cúp điện",
            "mất nước",
            "đình công", "biểu tình",
            "đóng cửa trường học",
            "nghỉ học", "làm việc từ xa",
            "quá tải", "chờ đợi lâu"
        ],
        "Prices": [
            "tăng giá", "leo thang giá",
            "giá xăng", "giá điện", "giá vé"
        ]
    },

    "T7_ROUTINE": {
        "Weather_Daily": [
            "dự báo thời tiết", "nắng nóng", "nắng nóng kéo dài",
            "rét đậm", "rét hại", "hạn hán", "nhiệt độ"
        ],
        "Sports_Routine": [
            "trận đấu", "chung kết", "bán kết",
            "giải đấu", "vô địch",
            "chiến thắng", "thất bại",
            "ghi bàn", "bàn thắng",
            "thẻ đỏ", "penalty",
            "chấn thương", "chuyển nhượng"
        ],
        "Lottery": [
            "xổ số", "xsmn", "xsmb", "vietlott", "quay thử"
        ]
    }
}

def get_flattened_keywords_by_group(group_key: str) -> list:
    """Returns a flat list of all keywords for a specific group (T1..T7)."""
    group_data = EVENT_KEYWORDS_VI.get(group_key, {})
    all_kws = []
    for subcat_kws in group_data.values():
        all_kws.extend(subcat_kws)
    return sorted(list(set(all_kws)))

def get_all_event_keywords() -> list:
    """Returns a flat list of every single keyword in the taxonomy."""
    all_kws = []
    for group in EVENT_KEYWORDS_VI.values():
        for subcat in group.values():
            all_kws.extend(subcat)
    return sorted(list(set(all_kws)))
