import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: ml1m/lfm1b/sample')
opt = parser.parse_args()
print(opt.dataset)
dataset = 'sorted_movielens.csv'
with open(dataset, "r") as f:
 reader = csv.DictReader(f)
 sess_clicks = {}
 sess_date = {}
 ctr = 0
 curid = -1
 curdate = None
 for data in reader:
  sessid = data['sessionID']
  #print(sessid)
  if curdate and curid != sessid:
   date = data['timestamp']
   # date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
   sess_date[curid] = date
  curid = sessid
  item = data['movieID']
  # curdate = ''
  curdate = data['timestamp']
  if sessid in sess_clicks:
   sess_clicks[sessid] += [item]
  else:
   sess_clicks[sessid] = [item]
  ctr += 1
 sess_date[curid] = date
 #print(sess_date)
 #print(sess_clicks)

for s in list(sess_clicks):
 if len(sess_clicks[s]) == 1:
  del sess_clicks[s]
  del sess_date[s]
#print(sess_date)
iid_counts = {}
for s in sess_clicks:
 seq = sess_clicks[s]
 for iid in seq:
  if iid in iid_counts:
   iid_counts[iid] += 1
  else:
   iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
 curseq = sess_clicks[s]
 filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
 if len(filseq) < 2:
  del sess_clicks[s]
  del sess_date[s]
 else:
  sess_clicks[s] = filseq

dates = list(sess_date.items())
#print(dates)
#maxdate = dates[0][1]


#for _, date in dates:
#if maxdate < date:
#maxdate = date

maxdate = 1537632338
splitdate = 0

splitdate = maxdate - 86400 * 2000   # the number of seconds for a day：86400


print('Splitting date', splitdate)  
tra_sess = list(filter(lambda x: int(x[1]) < splitdate, dates))
tes_sess = list(filter(lambda x: int(x[1]) > splitdate, dates))
#print(sorted_counts)
#print(tra_sess)
#print(tes_sess)

tra_sess = sorted(tra_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))  # [(session_id, timestamp), (), ]
print(len(tra_sess))  # 186670    # 7966257
print(len(tes_sess))  # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
 train_ids = []
 train_seqs = []
 train_dates = []
 item_ctr = 1
 for s, date in tra_sess:
  seq = sess_clicks[s]
  outseq = []
  for i in seq:
   if i in item_dict:
    outseq += [item_dict[i]]
   else:
    outseq += [item_ctr]
    item_dict[i] = item_ctr
    item_ctr += 1
  if len(outseq) < 2:  # Doesn't occur
   continue
  train_ids += [s]
  train_dates += [date]
  train_seqs += [outseq]
 print(item_ctr)  # 43098, 37484
 return train_ids, train_dates, train_seqs

def obtian_tes():
 test_ids = []
 test_seqs = []
 test_dates = []
 for s, date in tes_sess:
  seq = sess_clicks[s]
  outseq = []
  for i in seq:
   if i in item_dict:
    outseq += [item_dict[i]]
  if len(outseq) < 2:
   continue
  test_ids += [s]
  test_dates += [date]
  test_seqs += [outseq]
 return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):
 out_seqs = []
 out_dates = []
 labs = []
 ids = []
 for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
  for i in range(1, len(seq)):
   tar = seq[-i]
   labs += [tar]
   out_seqs += [seq[:-i]]
   out_dates += [date]
   ids += [id]
 return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
 all += len(seq)
for seq in tes_seqs:
 all += len(seq)
print('avg length: ', all / (len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('movielens'):
 os.makedirs('movielens')
pickle.dump(tra, open('movielens/train.txt', 'wb'))
pickle.dump(tes, open('movielens/test.txt', 'wb'))
pickle.dump(tra_seqs, open('movielens/all_train_seq.txt', 'wb'))

print('Done.')