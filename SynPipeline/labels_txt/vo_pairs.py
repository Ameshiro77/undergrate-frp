vo_pairs = [  # 放的是hoi
    (7, 8),
    (11, 13),
    (12, 19, 20, 21),
    (26, 28),
    (33, 42),
    (39, 42),
    (39, 40,42),
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
    (219, 218),
    (219, 223),
    (227, 231),
    (249, 251),
    (260, 259),
    (260, 261),
    (267, 377, 378),  # 用刀
    (258, 377, 378),
    (278, 280),
    (278, 277),
    (276, 377, 378),
    (285, 288),
    (297, 298),
    (309, 308),
    (309, 312),
    (331, 332),
    (353, 354),
    (359, 377, 378),
    (361, 360),
    (386, 388),
    (394, 395),
    (401, 400),
    (401, 377, 378),
    (422, 377, 378),
    (423, 424),
    (421, 425),
    (440, 444),
    (441, 377, 378),
    (443, 442),
    (446, 447),
    (454, 457),
    (460, 462),
    (465, 466),
    (471, 472),
    (476, 478),
    (480, 481),
    (484, 485),
    (484, 486),
    (490, 494),
    (507, 509),
    (508, 509),
    (517, 519),
    (518, 519),
    (524, 525),
    (535,537)
    (559, 560),
    (573, 574),
    (578, 583),
    (590, 593),
]

multi_hoi = [
    7,
    8,
    11,
    13,
    12,
    19,
    22,
    23,
    26,
    30,
    33,
    44,
    41,
    42,
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
    149,
    156,
    154,
    157,
    155,
    158,
    191,
    192,
    202,
    206,
    201,
    219,
    218,
    223,
    227,
    233,
    249,
    251,
    260,
    256,
    261,
    267,
    377,
    378,
    259,
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
    vos = []
    for hois in vo_pairs:
        for hoi in hois:
            if hoi not in vos:
                vos.append(hoi)
    print(vos)
