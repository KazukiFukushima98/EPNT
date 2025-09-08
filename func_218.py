import symengine
from symengine import Symbol

from hobotan import *
##計算重いから自己ループ削除

def symb(num, Species, MDcmp, cmp):

  #文字の設定
  moji_list = []
  N = []
  for n in range(num):
    mj = "I" * (n + 1)
    N.append(mj)
  #性能用
  for i in N:
    for s in Species:
      moji = "u_" + str(i) + s
      # Symbol(moji)
      moji_list.append(moji)

  lip = [1 + 3 * i for i in range(num)]
  lp = [x + 1 for x in lip]
  lip.append(lp[-1] + 2)

  lir = [1 + 3 * i for i in range(num)]
  lr = [x + 2 for x in lir]
  lir.append(lr[-1] + 2)

  st_list = []
  for i in lip[:-1]:
      for p, r in zip(lp, lr):
        moji = "t_" + str(p) + str(i)
        # Symbol(moji)
        moji_list.append(moji)
        st_list.append(moji)
        moji = "t_" + str(r) + str(i)
        # Symbol(moji)
        moji_list.append(moji)
        st_list.append(moji)

  for p, r in zip(lp, lr):
      moji = "t_" + str(p) + str(lip[-1])
      # Symbol(moji)
      moji_list.append(moji)
      st_list.append(moji)
      moji = "t_" + str(r) + str(lir[-1])
      # Symbol(moji)
      moji_list.append(moji)
      st_list.append(moji)

  sMD_list = []
  for c in range(MDcmp):
    moji_list.append("MD_" + str(c))
    sMD_list.append("MD_" + str(c))

  sX_list = []
  for n in range(num):
    mj = "I" * (n + 1)
    for c in range(cmp):
      moji_list.append("X_" + mj + str(c))
      sX_list.append("X_" + mj + str(c))

  Symbols = []
  local_vars = {}

  for s in moji_list:
    local_vars[s] = Symbol(s)
    Symbols.append(local_vars[s])

  t_list = []
  tlocal_vars = {}
  for s in st_list:
    tlocal_vars[s] = Symbol(s)
    t_list.append(tlocal_vars[s])

  MD_list = []
  MDlocal_vars = {}
  for s in sMD_list:
    MDlocal_vars[s] = Symbol(s)
    MD_list.append(MDlocal_vars[s])

  X_list = []
  xlocal_vars = {}
  for s in sX_list:
    xlocal_vars[s] = Symbol(s)
    X_list.append(xlocal_vars[s])

  lsum = [t_list, MD_list, X_list]

  return local_vars, Symbols, lsum, moji_list
