# -*- coding: utf-8 -*-
"""
4.4 定性演示：用现代文（人民日报 9801）训练的断句标点模型，给无标点古文做句读。
仅作定性展示（训练域 ≠ 目标域），不做定量声明。
"""
import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from decode_long import (load_vocab, doc_windows, predict_windows, merge_overlap,
                         encode_text)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "punct_best")

GUSHI = {
    "岳阳楼记（节选）": "庆历四年春滕子京谪守巴陵郡越明年政通人和百废具兴乃重修岳阳楼增其旧制刻唐贤今人诗赋于其上属予作文以记之予观夫巴陵胜状在洞庭一湖衔远山吞长江浩浩汤汤横无际涯朝晖夕阴气象万千此则岳阳楼之大观也前人之述备矣然则北通巫峡南极潇湘迁客骚人多会于此览物之情得无异乎若夫霪雨霏霏连月不开阴风怒号浊浪排空日星隐曜山岳潜形商旅不行樯倾楫摧薄暮冥冥虎啸猿啼登斯楼也则有去国怀乡忧谗畏讥满目萧然感极而悲者矣至若春和景明波澜不惊上下天光一碧万顷沙鸥翔集锦鳞游泳岸芷汀兰郁郁青青而或长烟一空皓月千里浮光跃金静影沉璧渔歌互答此乐何极登斯楼也则有心旷神怡宠辱偕忘把酒临风其喜洋洋者矣",
    "师说（节选）": "古之学者必有师师者所以传道受业解惑也人非生而知之者孰能无惑惑而不从师其为惑也终不解矣生乎吾前其闻道也固先乎吾吾从而师之生乎吾后其闻道也亦先乎吾吾从而师之吾师道也夫庸知其年之先后生于吾乎是故无贵无贱无长无少道之所存师之所存也",
    "爱莲说（节选）": "水陆草木之花可爱者甚蕃晋陶渊明独爱菊自李唐来世人甚爱牡丹予独爱莲之出淤泥而不染濯清涟而不妖中通外直不蔓不枝香远益清亭亭净植可远观而不可亵玩焉予谓菊花之隐逸者也牡丹花之富贵者也莲花之君子者也噫菊之爱陶后鲜有闻莲之爱同予者何人牡丹之爱宜乎众矣",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    vocab, id2lab = load_vocab()
    n_classes = len(vocab)

    out_lines = []
    for title, text in GUSHI.items():
        wins = doc_windows(len(text))
        prepared = []
        for s, e in wins:
            ids, char_pos = encode_text(tok, text[s:e])
            prepared.append((s, e, ids, char_pos))
        groups = [prepared[i:i + 8] for i in range(0, len(prepared), 8)]
        windows_pred = predict_windows(model, tok, id2lab, groups, device)
        merged = merge_overlap(windows_pred, len(text), n_classes)
        pred = merged.argmax(-1)
        rebuilt = "".join(text[i] + (id2lab[int(pred[i])] if id2lab[int(pred[i])] != "O" else "")
                          for i in range(len(text)))
        out_lines.append(f"【{title}】\n{rebuilt}\n")
    out = "\n".join(out_lines)
    with open(os.path.join(HERE, "results", "gushi_demo.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
