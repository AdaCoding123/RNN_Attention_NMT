# -*- coding:utf-8 -*-
f_cn = open('cn.txt', 'r', encoding='utf-8')
f_en = open('en.txt', 'r', encoding='utf-8')
cn_length = 0
en_length = 0
with open('train.tsv', 'w', encoding='utf-8') as f:
	for cn, en in zip(f_cn, f_en):
		f.write(cn.strip()+'\t'+'<bos>'+' '+en.strip()+' '+'<eos>\n')
		cn = len(cn.split())
		en = len(en.split())
		if cn > cn_length:
			cn_length = cn
		if en > en_length:
			en_length = en
f_cn.close()
f_en.close()
print('max CN length is {}, max EN length is {}'.format(cn_length, en_length))