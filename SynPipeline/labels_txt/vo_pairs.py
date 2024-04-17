vo_pairs = [  # 放的是hoi
    (7, 8),
    (11, 13),
    (12, 19, 20, 21),
    # (26, 28),
    (33, 42),
    (39, 42),
    (39, 40, 42),
    (48, 49),
    (49, 51),
    (49, 52),
    (61, 62),
    (57, 62),
    (109, 110),
    (140, 142),
    (147, 156, 154, 155),
    (153, 155, 156, 154),
    (191, 192),
    (202, 206),
    (202, 201),
    (209, 213),
    (219, 218),
    (219, 223),
    (227, 231),
    (249, 251),
    (260, 259),
    (261, 260),
    (267, 377, 378),  # 用刀
    (258, 377, 378),
    (278, 280),
    (278, 277),
    (276, 377, 378),
    (285, 288),
    (285, 286),
    (285, 289),
    (297, 298),
    (309, 308),
    (309, 312),
    (331, 332),
    (539, 545),
    (349, 350, 77),  # 用drier
    (349, 350, 113),
    (353, 354),
    (359, 377, 378),
    (361, 360),
    (386, 388),
    (394, 395),
    (401, 400),
    (401, 377, 378),
    (422, 377, 378),
    (424, 423),
    (424, 428),
    (421, 425),
    (440, 444),
    (441, 377, 378),
    (443, 442),
    (446, 447),
    (454, 457),
    (460, 462),
    (465, 466),
    (471, 472, 473),
    (468, 471, 472, 473),
    (472, 473),
    (476, 478),
    (480, 481, 482),
    (479, 480, 482),
    (481, 482),
    (477, 480, 481, 482),
    (484, 485),
    (484, 486),
    (490, 494),
    (494, 499),
    (507, 509),
    (508, 509),
    (517, 519),
    (518, 519),
    (524, 525),
    (535, 537),
    (559, 560),
    (573, 574),
    (578, 577, 583),
    (590, 593),
]

aux_verb_noun = { #得留空
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: " with hand",  # carry
    10: "",
    11: "",
    12: "",
    13: "",
    14: "",
    15: "",
    16: "",
    17: "",
    18: "",
    19: "",
    20: "",
    21: "",
    22: "",
    23: "",
    24: " with mouse",  # eat
    25: "",
    26: "",
    27: "",
    28: "",
    29: "",
    30: "",
    31: "",
    32: "",
    33: "",
    34: "",
    35: "",
    36: "",
    37: " with hand",  # hold
    38: "",
    39: "",
    40: "",
    41: "",
    42: " by looking close", #inspect
    43: "",
    44: "",
    45: " with feet",
    46: "",
    47: "",
    48: "",
    49: "",
    50: "",
    51: "",
    52: "",
    53: "",
    54: "",
    55: "",
    56: "",
    57: "",
    58: "",
    59: " with hand", #open
    60: "",
    61: "",
    62: "",
    63: "",
    64: "",
    65: "",
    66: "",
    67: "",
    68: "",
    69: "",
    70: "",
    71: "",
    72: "",
    73: "",
    74: "",
    75: "",
    76: " with tools", #repair
    77: "",
    78: "",
    79: "",
    80: "",
    81: "",
    82: "",
    83: "",
    84: "",
    85: "",
    86: "",
    87: "",
    88: "",
    89: "",
    90: " with nose",  # smell
    91: "",
    92: "",
    93: "",
    94: "",
    95: "",
    96: "",
    97: "",
    98: "",
    99: "",
    100: "",
    101: "",
    102: "",
    103: "",
    104: "",
    105: "",
    106: "",
    107: "",
    108: "",
    109: "",
    110: "",
    111: "",
    112: " with water",  # wash
    113: "",
    114: "",
    115: "",
    116: "",
    117: "",
}


multi_hoi = [
    7,
    8,
    11,
    13,
    12,
    19,
    20,
    21,
    26,
    28,
    33,
    42,
    39,
    40,
    48,
    49,
    51,
    52,
    61,
    62,
    57,
    109,
    110,
    140,
    142,
    147,
    156,
    154,
    155,
    153,
    191,
    192,
    202,
    206,
    201,
    219,
    218,
    223,
    227,
    231,
    249,
    251,
    260,
    259,
    261,
    267,
    377,
    378,
    258,
    278,
    280,
    277,
    276,
    285,
    288,
    297,
    298,
    309,
    308,
    312,
    331,
    332,
    353,
    354,
    359,
    361,
    360,
    386,
    388,
    394,
    395,
    401,
    400,
    422,
    423,
    424,
    421,
    425,
    440,
    444,
    441,
    443,
    442,
    446,
    447,
    454,
    457,
    460,
    462,
    465,
    466,
    471,
    472,
    476,
    478,
    480,
    481,
    484,
    485,
    486,
    490,
    494,
    507,
    509,
    508,
    517,
    519,
    518,
    524,
    525,
    535,
    537,
    559,
    560,
    573,
    574,
    578,
    583,
    590,
    593,
]

if __name__ == "__main__":
    # vos = []
    # for hois in vo_pairs:
    #     for hoi in hois:
    #         if hoi not in vos:
    #             vos.append(hoi)
    vos = {}
    for i in range(117):
        vos[i + 1] = ""
    print(vos)
