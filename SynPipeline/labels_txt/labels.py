# 保存生成的labels。


# (解析coco的id)
valid_obj_ids = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)

# {解析id：原始id}
original_labels_dict = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}

# (动词id,1~117)
valid_verb_ids = list(range(1, 118))

# verb to id
verb_to_id_dict = {
    "adjust": 1,
    "assemble": 2,
    "block": 3,
    "blow": 4,
    "board": 5,
    "break": 6,
    "brush_with": 7,
    "buy": 8,
    "carry": 9,
    "catch": 10,
    "chase": 11,
    "check": 12,
    "clean": 13,
    "control": 14,
    "cook": 15,
    "cut": 16,
    "cut_with": 17,
    "direct": 18,
    "drag": 19,
    "dribble": 20,
    "drink_with": 21,
    "drive": 22,
    "dry": 23,
    "eat": 24,
    "eat_at": 25,
    "exit": 26,
    "feed": 27,
    "fill": 28,
    "flip": 29,
    "flush": 30,
    "fly": 31,
    "greet": 32,
    "grind": 33,
    "groom": 34,
    "herd": 35,
    "hit": 36,
    "hold": 37,
    "hop_on": 38,
    "hose": 39,
    "hug": 40,
    "hunt": 41,
    "inspect": 42,
    "install": 43,
    "jump": 44,
    "kick": 45,
    "kiss": 46,
    "lasso": 47,
    "launch": 48,
    "lick": 49,
    "lie_on": 50,
    "lift": 51,
    "light": 52,
    "load": 53,
    "lose": 54,
    "make": 55,
    "milk": 56,
    "move": 57,
    "no_interaction": 58,
    "open": 59,
    "operate": 60,
    "pack": 61,
    "paint": 62,
    "park": 63,
    "pay": 64,
    "peel": 65,
    "pet": 66,
    "pick": 67,
    "pick_up": 68,
    "point": 69,
    "pour": 70,
    "pull": 71,
    "push": 72,
    "race": 73,
    "read": 74,
    "release": 75,
    "repair": 76,
    "ride": 77,
    "row": 78,
    "run": 79,
    "sail": 80,
    "scratch": 81,
    "serve": 82,
    "set": 83,
    "shear": 84,
    "sign": 85,
    "sip": 86,
    "sit_at": 87,
    "sit_on": 88,
    "slide": 89,
    "smell": 90,
    "spin": 91,
    "squeeze": 92,
    "stab": 93,
    "stand_on": 94,
    "stand_under": 95,
    "stick": 96,
    "stir": 97,
    "stop_at": 98,
    "straddle": 99,
    "swing": 100,
    "tag": 101,
    "talk_on": 102,
    "teach": 103,
    "text_on": 104,
    "throw": 105,
    "tie": 106,
    "toast": 107,
    "train": 108,
    "turn": 109,
    "type_on": 110,
    "walk": 111,
    "wash": 112,
    "watch": 113,
    "wave": 114,
    "wear": 115,
    "wield": 116,
    "zip": 117,
}

# obj to id (解析的obj序号)
obj_to_id_dict = {
    "airplane": 1,
    "apple": 2,
    "backpack": 3,
    "banana": 4,
    "baseball_bat": 5,
    "baseball_glove": 6,
    "bear": 7,
    "bed": 8,
    "bench": 9,
    "bicycle": 10,
    "bird": 11,
    "boat": 13,
    "book": 14,
    "bottle": 15,
    "bowl": 16,
    "broccoli": 17,
    "bus": 18,
    "cake": 19,
    "car": 20,
    "carrot": 21,
    "cat": 22,
    "cell_phone": 23,
    "chair": 24,
    "clock": 25,
    "couch": 27,
    "cow": 28,
    "cup": 31,
    "dining_table": 32,
    "dog": 33,
    "donut": 34,
    "elephant": 35,
    "fire_hydrant": 36,
    "fork": 37,
    "frisbee": 38,
    "giraffe": 39,
    "hair_drier": 40,
    "handbag": 41,
    "horse": 42,
    "hot_dog": 43,
    "keyboard": 44,
    "kite": 46,
    "knife": 47,
    "laptop": 48,
    "microwave": 49,
    "motorcycle": 50,
    "mouse": 51,
    "orange": 52,
    "oven": 53,
    "parking_meter": 54,
    "person": 55,
    "pizza": 56,
    "potted_plant": 57,
    "refrigerator": 58,
    "remote": 59,
    "sandwich": 60,
    "scissors": 61,
    "sheep": 62,
    "sink": 63,
    "skateboard": 64,
    "skis": 65,
    "snowboard": 67,
    "spoon": 70,
    "sports_ball": 72,
    "stop_sign": 73,
    "suitcase": 74,
    "surfboard": 75,
    "teddy_bear": 76,
    "tennis_racket": 77,
    "tie": 78,
    "toaster": 79,
    "toilet": 80,
    "toothbrush": 81,
    "traffic_light": 82,
    "train": 84,
    "truck": 85,
    "tv": 86,
    "umbrella": 87,
    "vase": 88,
    "wine_glass": 89,
    "zebra": 90,
}

# hoi(verb,obj) to id (序号代替)
hoi_to_id_dict = {
    (5, 1): 1,
    (18, 1): 2,
    (26, 1): 3,
    (31, 1): 4,
    (42, 1): 5,
    (53, 1): 6,
    (77, 1): 7,
    (88, 1): 8,
    (112, 1): 9,
    (58, 1): 10,
    (9, 10): 11,
    (37, 10): 12,
    (42, 10): 13,
    (44, 10): 14,
    (38, 10): 15,
    (63, 10): 16,
    (72, 10): 17,
    (76, 10): 18,
    (77, 10): 19,
    (88, 10): 20,
    (99, 10): 21,
    (111, 10): 22,
    (112, 10): 23,
    (58, 10): 24,
    (11, 11): 25,
    (27, 11): 26,
    (37, 11): 27,
    (66, 11): 28,
    (75, 11): 29,
    (113, 11): 30,
    (58, 11): 31,
    (5, 13): 32,
    (22, 13): 33,
    (26, 13): 34,
    (42, 13): 35,
    (44, 13): 36,
    (48, 13): 37,
    (76, 13): 38,
    (77, 13): 39,
    (78, 13): 40,
    (80, 13): 41,
    (88, 13): 42,
    (94, 13): 43,
    (106, 13): 44,
    (112, 13): 45,
    (58, 13): 46,
    (9, 15): 47,
    (21, 15): 48,
    (37, 15): 49,
    (42, 15): 50,
    (49, 15): 51,
    (59, 15): 52,
    (70, 15): 53,
    (58, 15): 54,
    (5, 18): 55,
    (18, 18): 56,
    (22, 18): 57,
    (26, 18): 58,
    (42, 18): 59,
    (53, 18): 60,
    (77, 18): 61,
    (88, 18): 62,
    (112, 18): 63,
    (114, 18): 64,
    (58, 18): 65,
    (5, 20): 66,
    (18, 20): 67,
    (22, 20): 68,
    (39, 20): 69,
    (42, 20): 70,
    (44, 20): 71,
    (53, 20): 72,
    (63, 20): 73,
    (77, 20): 74,
    (112, 20): 75,
    (58, 20): 76,
    (23, 22): 77,
    (27, 22): 78,
    (37, 22): 79,
    (40, 22): 80,
    (46, 22): 81,
    (66, 22): 82,
    (81, 22): 83,
    (112, 22): 84,
    (11, 22): 85,
    (58, 22): 86,
    (9, 24): 87,
    (37, 24): 88,
    (50, 24): 89,
    (88, 24): 90,
    (94, 24): 91,
    (58, 24): 92,
    (9, 27): 93,
    (50, 27): 94,
    (88, 27): 95,
    (58, 27): 96,
    (27, 28): 97,
    (35, 28): 98,
    (37, 28): 99,
    (40, 28): 100,
    (46, 28): 101,
    (47, 28): 102,
    (56, 28): 103,
    (66, 28): 104,
    (77, 28): 105,
    (111, 28): 106,
    (58, 28): 107,
    (13, 32): 108,
    (25, 32): 109,
    (87, 32): 110,
    (58, 32): 111,
    (9, 33): 112,
    (23, 33): 113,
    (27, 33): 114,
    (34, 33): 115,
    (37, 33): 116,
    (39, 33): 117,
    (40, 33): 118,
    (42, 33): 119,
    (46, 33): 120,
    (66, 33): 121,
    (79, 33): 122,
    (81, 33): 123,
    (99, 33): 124,
    (108, 33): 125,
    (111, 33): 126,
    (112, 33): 127,
    (11, 33): 128,
    (58, 33): 129,
    (27, 42): 130,
    (34, 42): 131,
    (37, 42): 132,
    (40, 42): 133,
    (44, 42): 134,
    (46, 42): 135,
    (53, 42): 136,
    (38, 42): 137,
    (66, 42): 138,
    (73, 42): 139,
    (77, 42): 140,
    (79, 42): 141,
    (99, 42): 142,
    (108, 42): 143,
    (111, 42): 144,
    (112, 42): 145,
    (58, 42): 146,
    (37, 50): 147,
    (42, 50): 148,
    (44, 50): 149,
    (38, 50): 150,
    (63, 50): 151,
    (72, 50): 152,
    (73, 50): 153,
    (77, 50): 154,
    (88, 50): 155,
    (99, 50): 156,
    (109, 50): 157,
    (111, 50): 158,
    (112, 50): 159,
    (58, 50): 160,
    (9, 55): 161,
    (32, 55): 162,
    (37, 55): 163,
    (40, 55): 164,
    (46, 55): 165,
    (93, 55): 166,
    (101, 55): 167,
    (103, 55): 168,
    (49, 55): 169,
    (58, 55): 170,
    (9, 57): 171,
    (37, 57): 172,
    (39, 57): 173,
    (58, 57): 174,
    (9, 62): 175,
    (27, 62): 176,
    (35, 62): 177,
    (37, 62): 178,
    (40, 62): 179,
    (46, 62): 180,
    (66, 62): 181,
    (77, 62): 182,
    (84, 62): 183,
    (111, 62): 184,
    (112, 62): 185,
    (58, 62): 186,
    (5, 84): 187,
    (22, 84): 188,
    (26, 84): 189,
    (53, 84): 190,
    (77, 84): 191,
    (88, 84): 192,
    (112, 84): 193,
    (58, 84): 194,
    (14, 86): 195,
    (76, 86): 196,
    (113, 86): 197,
    (58, 86): 198,
    (8, 2): 199,
    (16, 2): 200,
    (24, 2): 201,
    (37, 2): 202,
    (42, 2): 203,
    (65, 2): 204,
    (67, 2): 205,
    (90, 2): 206,
    (112, 2): 207,
    (58, 2): 208,
    (9, 3): 209,
    (37, 3): 210,
    (42, 3): 211,
    (59, 3): 212,
    (115, 3): 213,
    (58, 3): 214,
    (8, 4): 215,
    (9, 4): 216,
    (16, 4): 217,
    (24, 4): 218,
    (37, 4): 219,
    (42, 4): 220,
    (65, 4): 221,
    (67, 4): 222,
    (90, 4): 223,
    (58, 4): 224,
    (6, 5): 225,
    (9, 5): 226,
    (37, 5): 227,
    (85, 5): 228,
    (100, 5): 229,
    (105, 5): 230,
    (116, 5): 231,
    (58, 5): 232,
    (37, 6): 233,
    (115, 6): 234,
    (58, 6): 235,
    (27, 7): 236,
    (41, 7): 237,
    (113, 7): 238,
    (58, 7): 239,
    (13, 8): 240,
    (50, 8): 241,
    (88, 8): 242,
    (58, 8): 243,
    (42, 9): 244,
    (50, 9): 245,
    (88, 9): 246,
    (58, 9): 247,
    (9, 14): 248,
    (37, 14): 249,
    (59, 14): 250,
    (74, 14): 251,
    (58, 14): 252,
    (37, 16): 253,
    (97, 16): 254,
    (112, 16): 255,
    (49, 16): 256,
    (58, 16): 257,
    (16, 17): 258,
    (24, 17): 259,
    (37, 17): 260,
    (90, 17): 261,
    (97, 17): 262,
    (112, 17): 263,
    (58, 17): 264,
    (4, 19): 265,
    (9, 19): 266,
    (16, 19): 267,
    (24, 19): 268,
    (37, 19): 269,
    (52, 19): 270,
    (55, 19): 271,
    (68, 19): 272,
    (58, 19): 273,
    (9, 21): 274,
    (15, 21): 275,
    (16, 21): 276,
    (24, 21): 277,
    (37, 21): 278,
    (65, 21): 279,
    (90, 21): 280,
    (97, 21): 281,
    (112, 21): 282,
    (58, 21): 283,
    (9, 23): 284,
    (37, 23): 285,
    (74, 23): 286,
    (76, 23): 287,
    (102, 23): 288,
    (104, 23): 289,
    (58, 23): 290,
    (12, 25): 291,
    (37, 25): 292,
    (76, 25): 293,
    (83, 25): 294,
    (58, 25): 295,
    (9, 31): 296,
    (21, 31): 297,
    (37, 31): 298,
    (42, 31): 299,
    (70, 31): 300,
    (86, 31): 301,
    (90, 31): 302,
    (28, 31): 303,
    (112, 31): 304,
    (58, 31): 305,
    (8, 34): 306,
    (9, 34): 307,
    (24, 34): 308,
    (37, 34): 309,
    (55, 34): 310,
    (68, 34): 311,
    (90, 34): 312,
    (58, 34): 313,
    (27, 35): 314,
    (37, 35): 315,
    (39, 35): 316,
    (40, 35): 317,
    (46, 35): 318,
    (38, 35): 319,
    (66, 35): 320,
    (77, 35): 321,
    (111, 35): 322,
    (112, 35): 323,
    (113, 35): 324,
    (58, 35): 325,
    (40, 36): 326,
    (42, 36): 327,
    (59, 36): 328,
    (62, 36): 329,
    (58, 36): 330,
    (37, 37): 331,
    (51, 37): 332,
    (96, 37): 333,
    (49, 37): 334,
    (112, 37): 335,
    (58, 37): 336,
    (3, 38): 337,
    (10, 38): 338,
    (37, 38): 339,
    (91, 38): 340,
    (105, 38): 341,
    (58, 38): 342,
    (27, 39): 343,
    (46, 39): 344,
    (66, 39): 345,
    (77, 39): 346,
    (113, 39): 347,
    (58, 39): 348,
    (37, 40): 349,
    (60, 40): 350,
    (76, 40): 351,
    (58, 40): 352,
    (9, 41): 353,
    (37, 41): 354,
    (42, 41): 355,
    (58, 41): 356,
    (9, 43): 357,
    (15, 43): 358,
    (16, 43): 359,
    (24, 43): 360,
    (37, 43): 361,
    (55, 43): 362,
    (58, 43): 363,
    (9, 44): 364,
    (13, 44): 365,
    (37, 44): 366,
    (110, 44): 367,
    (58, 44): 368,
    (2, 46): 369,
    (9, 46): 370,
    (31, 46): 371,
    (37, 46): 372,
    (42, 46): 373,
    (48, 46): 374,
    (71, 46): 375,
    (58, 46): 376,
    (17, 47): 377,
    (37, 47): 378,
    (96, 47): 379,
    (112, 47): 380,
    (116, 47): 381,
    (49, 47): 382,
    (58, 47): 383,
    (37, 48): 384,
    (59, 48): 385,
    (74, 48): 386,
    (76, 48): 387,
    (110, 48): 388,
    (58, 48): 389,
    (13, 49): 390,
    (59, 49): 391,
    (60, 49): 392,
    (58, 49): 393,
    (14, 51): 394,
    (37, 51): 395,
    (76, 51): 396,
    (58, 51): 397,
    (8, 52): 398,
    (16, 52): 399,
    (24, 52): 400,
    (37, 52): 401,
    (42, 52): 402,
    (65, 52): 403,
    (67, 52): 404,
    (92, 52): 405,
    (112, 52): 406,
    (58, 52): 407,
    (13, 53): 408,
    (37, 53): 409,
    (42, 53): 410,
    (59, 53): 411,
    (76, 53): 412,
    (60, 53): 413,
    (58, 53): 414,
    (12, 54): 415,
    (64, 54): 416,
    (76, 54): 417,
    (58, 54): 418,
    (8, 56): 419,
    (9, 56): 420,
    (15, 56): 421,
    (16, 56): 422,
    (24, 56): 423,
    (37, 56): 424,
    (55, 56): 425,
    (68, 56): 426,
    (89, 56): 427,
    (90, 56): 428,
    (58, 56): 429,
    (13, 58): 430,
    (37, 58): 431,
    (57, 58): 432,
    (59, 58): 433,
    (58, 58): 434,
    (37, 59): 435,
    (69, 59): 436,
    (100, 59): 437,
    (58, 59): 438,
    (9, 60): 439,
    (15, 60): 440,
    (16, 60): 441,
    (24, 60): 442,
    (37, 60): 443,
    (55, 60): 444,
    (58, 60): 445,
    (17, 61): 446,
    (37, 61): 447,
    (59, 61): 448,
    (58, 61): 449,
    (13, 63): 450,
    (76, 63): 451,
    (112, 63): 452,
    (58, 63): 453,
    (9, 64): 454,
    (29, 64): 455,
    (33, 64): 456,
    (37, 64): 457,
    (44, 64): 458,
    (68, 64): 459,
    (77, 64): 460,
    (88, 64): 461,
    (94, 64): 462,
    (58, 64): 463,
    (1, 65): 464,
    (9, 65): 465,
    (37, 65): 466,
    (42, 65): 467,
    (44, 65): 468,
    (68, 65): 469,
    (76, 65): 470,
    (77, 65): 471,
    (94, 65): 472,
    (115, 65): 473,
    (58, 65): 474,
    (1, 67): 475,
    (9, 67): 476,
    (33, 67): 477,
    (37, 67): 478,
    (44, 67): 479,
    (77, 67): 480,
    (94, 67): 481,
    (115, 67): 482,
    (58, 67): 483,
    (37, 70): 484,
    (49, 70): 485,
    (112, 70): 486,
    (86, 70): 487,
    (58, 70): 488,
    (3, 72): 489,
    (9, 72): 490,
    (10, 72): 491,
    (20, 72): 492,
    (36, 72): 493,
    (37, 72): 494,
    (42, 72): 495,
    (45, 72): 496,
    (68, 72): 497,
    (82, 72): 498,
    (85, 72): 499,
    (91, 72): 500,
    (105, 72): 501,
    (58, 72): 502,
    (37, 73): 503,
    (95, 73): 504,
    (98, 73): 505,
    (58, 73): 506,
    (9, 74): 507,
    (19, 74): 508,
    (37, 74): 509,
    (40, 74): 510,
    (53, 74): 511,
    (59, 74): 512,
    (61, 74): 513,
    (68, 74): 514,
    (117, 74): 515,
    (58, 74): 516,
    (9, 75): 517,
    (19, 75): 518,
    (37, 75): 519,
    (42, 75): 520,
    (44, 75): 521,
    (50, 75): 522,
    (53, 75): 523,
    (77, 75): 524,
    (94, 75): 525,
    (88, 75): 526,
    (112, 75): 527,
    (58, 75): 528,
    (9, 76): 529,
    (37, 76): 530,
    (40, 76): 531,
    (46, 76): 532,
    (58, 76): 533,
    (9, 77): 534,
    (37, 77): 535,
    (42, 77): 536,
    (100, 77): 537,
    (58, 77): 538,
    (1, 78): 539,
    (16, 78): 540,
    (37, 78): 541,
    (42, 78): 542,
    (71, 78): 543,
    (106, 78): 544,
    (115, 78): 545,
    (58, 78): 546,
    (37, 79): 547,
    (60, 79): 548,
    (76, 79): 549,
    (58, 79): 550,
    (13, 80): 551,
    (30, 80): 552,
    (59, 80): 553,
    (76, 80): 554,
    (88, 80): 555,
    (94, 80): 556,
    (112, 80): 557,
    (58, 80): 558,
    (7, 81): 559,
    (37, 81): 560,
    (112, 81): 561,
    (58, 81): 562,
    (43, 82): 563,
    (76, 82): 564,
    (95, 82): 565,
    (98, 82): 566,
    (58, 82): 567,
    (18, 85): 568,
    (22, 85): 569,
    (42, 85): 570,
    (53, 85): 571,
    (76, 85): 572,
    (77, 85): 573,
    (88, 85): 574,
    (112, 85): 575,
    (58, 85): 576,
    (9, 87): 577,
    (37, 87): 578,
    (54, 87): 579,
    (59, 87): 580,
    (76, 87): 581,
    (83, 87): 582,
    (95, 87): 583,
    (58, 87): 584,
    (37, 88): 585,
    (55, 88): 586,
    (62, 88): 587,
    (58, 88): 588,
    (28, 89): 589,
    (37, 89): 590,
    (86, 89): 591,
    (107, 89): 592,
    (49, 89): 593,
    (112, 89): 594,
    (58, 89): 595,
    (27, 90): 596,
    (37, 90): 597,
    (66, 90): 598,
    (113, 90): 599,
    (58, 90): 600,
}
