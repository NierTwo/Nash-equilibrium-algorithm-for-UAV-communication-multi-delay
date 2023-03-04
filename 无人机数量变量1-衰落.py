import random

import matplotlib.pyplot as mp
import numpy as np
from math import *
from random import *
import networkx as nx
for nomean in range(1):
    mp.rcParams['font.sans-serif'] = ['KaiTi']

    def backhaul_remake2(l,U_sum,S_sum,B,Uxy,C):                                    ###l,U_sum,S_sum,S_sum,B为自定义的常数,Uxy,C为数组
        B1xy=[0,0]
        Uxy=sorted(Uxy,key=lambda x:min([np.linalg.norm([x[0]-B1xy[0],x[1]-B1xy[1]])]),reverse=True)

        Ux=[i[0] for i in Uxy]
        Uy=[i[1] for i in Uxy]
        U_connect=[U_sum for i in range(U_sum)]    #初始化连接基站，编号为(U_sum)

        UBxy=Uxy+[B1xy]



        def PP_down():
            p_down=[[i] for i in range(U_sum)]
            for i in range(U_sum):
                o=i
                while(1):
                    p_down[i].append(U_connect[o])
                    o=U_connect[o]
                    if o==U_sum or o==U_sum+1 or o==U_sum+2 or o==U_sum+3:
                        break
            return p_down


        def PP_up():
            p_up=[[] for i in range(U_sum)]
            for i in range(U_sum):
                for j in range(U_sum):
                     if i in P_down[j]:
                         p_up[i].append(j)
            return p_up


        c=299792458
        f=2000000000
        a=9.6
        b=0.28
        h=2500
        los=1
        nlos=20
        v=8                                         #每个包的比特数
        throughout=[0 for i in range(U_sum)]        #每个无人机的吞吐量


        def angle(x, y):
            l = np.linalg.norm([x[0] - y[0], x[1] - y[1]])
            Angle = atan(h / l)
            return Angle

        def Plos(x, y):
            plos = 1 / (1 + a * exp(-b * (angle(x, y) * 180 / pi - a)))
            return plos

        def Loss1(x, y):
            loss1 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + los
            return loss1

        def Loss2(x, y):
            loss2 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + nlos
            return loss2

        def ULoss(x, y):
            uloss = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]])) + 20 * log10(f) + 20 * log10(
                4 * pi / c) + los
            return uloss

        def MLoss(x, y):
            mloss = Plos(x, y) * Loss1(x, y) + (1 - Plos(x, y)) * Loss2(x, y) - 1.85 * (angle(x,y) - 1.41)
            return mloss

        def USNR(x, y):
            usnr = 10 ** (P / 10) / 10 ** (ULoss(x, y) / 10) / 1e-9
            return usnr

        def MSNR(x, y):
            msnr = 10 ** (P / 10) / 10 ** (MLoss(x, y) / 10) / 1e-9
            return msnr

        def UADR(x, y):
            uadr = B * log2(1 + USNR(x, y))  # 无人机跳边的可到达数据量
            return uadr

        def MADR(x, y):
            adr = B * log2(1 + MSNR(x, y))  # 无人机跳边的可到达数据量
            return adr


        def R_min(x):                                               #评价函数U的分子
            hop_R=[]
            for i in P_down[x]:
                if P_down[x].index(i)==len(P_down[x])-2:
                    break
                r=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])
                hop_R.append(r)
            if P_down[x][-1]==U_sum:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B1xy))
            r=min(hop_R)
            return r


        def delay_cost(x):                                          #绝对评价函数U的分母
            r=0
            for i in P_down[x]:
                fai=0
                for j in P_up[i]:
                    fai+=C[j]
                if P_down[x].index(i)==len(P_down[x])-2:
                    if P_down[x][-1] == U_sum:
                        u=MADR(Uxy[i],B1xy)/v
                    if u<fai:
                        return float("inf")
                    r += fai / (2 * u * (u - fai)) + 1 / u
                    break
                else:
                    u=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])/v
                    if u<fai:
                        return float("inf")
                r+=fai/(2*u*(u-fai))+1/u
            return r


        def delay_cost2(x):                                          #相对评价函数U的分母
            r=0
            for i in P_down[x]:
                fai=0
                for j in P_up[i]:
                    fai+=C[j]
                if P_down[x].index(i)==len(P_down[x])-2:
                    if P_down[x][-1] == U_sum:
                        u=MADR(Uxy[i],B1xy)/v
                    if u<fai:
                        return 1
                    r += fai / (2 * u * (u - fai)) + 1 / u
                    break
                else:
                    u=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])/v
                    if u<fai:
                        return 1
                r+=fai/(2*u*(u-fai))+1/u
            return r


        def U(x):
            u=R_min(x)**0.5/delay_cost(x)**0.3
            return u


        def U2(x):
            u=R_min(x)**0.5/delay_cost2(x)**0.3
            return u


        U_order=[i for i in range(U_sum)]                        #Uorder为去除固定链路无人机的剩余无人机单元
        U_order0=[]                                              #为固定链路无人机单元集合


        for i in range(U_sum):                                   #预处理，增大信息传输通道
            P_down=PP_down()
            P_up=PP_up()
            for j in range(U_sum,U_sum+S_sum):
                begin_connect=U_connect[i]
                begin_U=U2(i)
                U_connect[i]=j
                P_down=PP_down()
                P_up=PP_up()
                if delay_cost(i)!=float("inf") and U2(i)>begin_U:
                    continue
                else:
                    U_connect[i]=begin_connect
            P_down=PP_down()
            P_up=PP_up()
            if delay_cost(i)!=float("inf"):
                throughout[i]=C[i]
                U_order.remove(i)
                U_order0.append(i)


        deallist=[]
        while(1):                                                                       #一级均衡
            for i in U_order:
                for j in range(len(UBxy)):
                    P_down = PP_down()
                    P_up = PP_up()
                    if i!=j and j not in P_up[i]:
                        begin_U_connect=U_connect[i]
                        U_begin = U(i)
                        U_connect[i] = j
                        P_down = PP_down()
                        P_up=PP_up()
                        if U(i)>U_begin:
                            continue
                        else:
                            U_connect[i]=begin_U_connect
                            continue
                if U_connect[i]==U_sum and MADR(Uxy[i],UBxy[U_connect[i]])/v<C[i]:
                    throughout[i]=MADR(Uxy[i],UBxy[U_connect[i]])/v
                else:
                    throughout[i]=C[i]

            if U_connect in deallist:
                if U_connect==deallist[-1]:
                    print('纳什收敛')
                else:
                    print('陷入循环')
                break
            deallist.append([i for i in U_connect])


        U_order2=[]                                                                     #剩余没有链路的无人机
        for i in U_order:
            if U_connect[i] >= U_sum:
                U_order2.append(i)


        ###处理剩余无人机的操作代码，首先遍历U_order也就是与基站相连接的无人机，然后可以求出每个入口的剩余可用数据速度，根据这个速度进行选择，在入口无人机的P_up中选择满足要求的即可###
        C_remain=[]
        P_down=PP_down()
        P_up=PP_up()
        for i in U_order0:
            u=MADR(Uxy[i],UBxy[U_connect[i]])/v
            fai=0
            for j in P_up[i]:
                fai+=C[j]
            c_remain=u-fai
            C_remain.append([i,c_remain])                                               #入口无人机的剩余数据量与每条入口链路进行绑定
        C_remain=sorted(C_remain,key=lambda x:x[1],reverse=True)



        for i in U_order2:
            for j in range(U_sum,U_sum+S_sum):
                begin_c=MADR(Uxy[i],UBxy[U_connect[i]])/v
                if MADR(Uxy[i],UBxy[j])/v>begin_c:
                    U_connect[i]=j
            throughout[i]=MADR(Uxy[i],UBxy[U_connect[i]])/v
            for j in C_remain:
                begin_connect=U_connect[i]
                if MADR(Uxy[i], UBxy[U_connect[i]])/v < j[1]:
                    if j[1] <= UADR(Uxy[i], Uxy[j[0]])/v:
                        U_connect[i]=j[0]
                        C_remain.remove(j)
                        throughout[i]=j[1]
                        break
                    else:
                        for z in P_up[j[0]]:
                            if j[1] <= UADR(Uxy[i],Uxy[z])/v:
                                U_connect[i]=z
                                C_remain.remove(j)
                                throughout[i]=j[1]
                                break

        # print(f"总数据传输速率为:{sum(throughout)}")
        # print(throughout)

        # mp.figure(3)
        # mp.xlim(0,l)
        # mp.ylim(0,l)
        # mp.plot(Ux,Uy,'^',color='red',markersize=10,label='无人机')
        # mp.plot(B1xy[0],B1xy[1],'o',color='black',markersize=10,label='基站')
        # BUX=[i[0] for i in UBxy]
        # BUY=[i[1] for i in UBxy]
        # for i in range(U_sum):
        #     mp.arrow(BUX[i],BUY[i],BUX[U_connect[i]]-BUX[i],BUY[U_connect[i]]-BUY[i],head_length=100,width=5,length_includes_head=True,fc='blue',ec='blue',overhang=0.5)
        # mp.legend()
        return throughout





    ######################################################################################################################################################################
    ######################################################################################################################################################################
    ######################################################################################################################################################################



    def backhaul_remake1_recycle(l,U_sum,S_sum,B,Uxy,C):
        B1xy=[0,0]
        B2xy=[0,l]
        B3xy=[l,0]
        B4xy=[l,l]
        Ux=[i[0] for i in Uxy]
        Uy=[i[1] for i in Uxy]
        U_connect=[U_sum for i in range(U_sum)]    #初始化连接基站，编号为(U_sum)

        UBxy=Uxy+[B1xy,B2xy,B3xy,B4xy]



        def PP_down():
            p_down=[[i] for i in range(U_sum)]
            for i in range(U_sum):
                o=i
                while(1):
                    p_down[i].append(U_connect[o])
                    o=U_connect[o]
                    if o==U_sum or o==U_sum+1 or o==U_sum+2 or o==U_sum+3:
                        break
            return p_down


        def PP_up():
            p_up=[[] for i in range(U_sum)]
            for i in range(U_sum):
                for j in range(U_sum):
                     if i in P_down[j]:
                         p_up[i].append(j)
            return p_up


        c=299792458
        f=2000000000
        a=9.6
        b=0.28
        h=2500
        los=1
        nlos=20
        v=8                                         #每个包的比特数
        throughout=[0 for i in range(U_sum)]        #每个无人机的吞吐量


        def angle(x, y):
            l = np.linalg.norm([x[0] - y[0], x[1] - y[1]])
            Angle = atan(h / l)
            return Angle

        def Plos(x, y):
            plos = 1 / (1 + a * exp(-b * (angle(x, y) * 180 / pi - a)))
            return plos

        def Loss1(x, y):
            loss1 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + los
            return loss1

        def Loss2(x, y):
            loss2 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + nlos
            return loss2

        def ULoss(x, y):
            uloss = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]])) + 20 * log10(f) + 20 * log10(
                4 * pi / c) + los
            return uloss

        def MLoss(x, y):
            mloss = Plos(x, y) * Loss1(x, y) + (1 - Plos(x, y)) * Loss2(x, y) - 1.85 * (angle(x,y) - 1.41)
            return mloss

        def USNR(x, y):
            usnr = 10 ** (P / 10) / 10 ** (ULoss(x, y) / 10) / 1e-9
            return usnr

        def MSNR(x, y):
            msnr = 10 ** (P / 10) / 10 ** (MLoss(x, y) / 10) / 1e-9
            return msnr

        def UADR(x, y):
            uadr = B * log2(1 + USNR(x, y))  # 无人机跳边的可到达数据量
            return uadr

        def MADR(x, y):
            adr = B * log2(1 + MSNR(x, y))  # 无人机跳边的可到达数据量
            return adr


        def R_min(x):                                               #评价函数U的分子
            hop_R=[]
            for i in P_down[x]:
                if P_down[x].index(i)==len(P_down[x])-2:
                    break
                r=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])
                hop_R.append(r)
            if P_down[x][-1]==U_sum:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B1xy))
            elif P_down[x][-1]==U_sum+1:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B2xy))
            elif P_down[x][-1]==U_sum+2:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B3xy))
            elif P_down[x][-1]==U_sum+3:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B4xy))
            r=min(hop_R)
            return r


        def delay_cost(x):                                          #绝对评价函数U的分母
            r=0
            for i in P_down[x]:
                fai=0
                for j in P_up[i]:
                    fai+=C[j]
                if P_down[x].index(i)==len(P_down[x])-2:
                    if P_down[x][-1] == U_sum:
                        u=MADR(Uxy[i],B1xy)/v
                    elif P_down[x][-1] == U_sum+1:
                        u=MADR(Uxy[i],B2xy)/v
                    elif P_down[x][-1] == U_sum+2:
                        u=MADR(Uxy[i],B3xy)/v
                    elif P_down[x][-1] == U_sum + 3:
                        u = MADR(Uxy[i], B4xy) / v
                    if u<fai:
                        return float("inf")
                    r += fai / (2 * u * (u - fai)) + 1 / u
                    break
                else:
                    u=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])/v
                    if u<fai:
                        return float("inf")
                r+=fai/(2*u*(u-fai))+1/u
            return r


        def U(x):
            u=R_min(x)**0.5/delay_cost(x)**0.3
            return u


        deallist=[]


        while(1):
            for i in range(U_sum):
                for j in range(U_sum+S_sum):
                    P_down = PP_down()
                    P_up = PP_up()
                    if i!=j and j not in P_up[i]:
                        begin_U_connect=U_connect[i]
                        U_begin = U(i)
                        U_connect[i] = j
                        P_down = PP_down()
                        P_up=PP_up()
                        if U(i)>U_begin:
                            continue
                        else:
                            U_connect[i]=begin_U_connect
                            continue
                if U_connect[i]==U_sum and MADR(Uxy[i],UBxy[U_connect[i]])/v<C[i]:
                    throughout[i]=MADR(Uxy[i],UBxy[U_connect[i]])/v
                else:
                    throughout[i]=C[i]

            if U_connect in deallist:
                if U_connect==deallist[-1]:
                    print('纳什收敛')
                else:
                    print('陷入循环')
                break
            deallist.append([i for i in U_connect])

        U_order0 = []
        P_down = PP_down()
        P_up = PP_up()
        for i in range(U_sum):
            if U_connect[i] >= U_sum:
                U_order0.append(i)

        U_order2 = []  # 剩余没有链路的无人机
        for i in range(U_sum):
            if U_connect[i] >= U_sum and P_up[i] == [i]:
                U_order2.append(i)

        ###处理剩余无人机的操作代码，首先遍历U_order也就是与基站相连接的无人机，然后可以求出每个入口的剩余可用数据速度，根据这个速度进行选择，在入口无人机的P_up中选择满足要求的即可###
        C_remain = []
        for i in U_order0:
            u = MADR(Uxy[i], UBxy[U_connect[i]]) / v
            fai = 0
            for j in P_up[i]:
                fai += C[j]
            c_remain = u - fai
            C_remain.append([i, c_remain])  # 入口无人机的剩余数据量与每条入口链路进行绑定
        C_remain = sorted(C_remain, key=lambda x: x[1], reverse=True)

        for i in U_order2:
            for j in range(U_sum, U_sum + S_sum):
                begin_c = MADR(Uxy[i], UBxy[U_connect[i]]) / v
                if MADR(Uxy[i], UBxy[j]) / v > begin_c:
                    U_connect[i] = j
            throughout[i] = min(MADR(Uxy[i], UBxy[U_connect[i]]) / v, C[i])
            for j in C_remain:
                begin_connect = U_connect[i]
                if MADR(Uxy[i], UBxy[U_connect[i]]) / v < j[1]:
                    if j[1] <= UADR(Uxy[i], Uxy[j[0]]) / v:
                        U_connect[i] = j[0]
                        C_remain.remove(j)
                        throughout[i] = j[1]
                        break
                    else:
                        for z in P_up[j[0]]:
                            if j[1] <= UADR(Uxy[i], Uxy[z]) / v:
                                U_connect[i] = z
                                C_remain.remove(j)
                                throughout[i] = j[1]
                                break
        return throughout



    ######################################################################################################################################################################
    ######################################################################################################################################################################
    ######################################################################################################################################################################





    def backhaul_remake1(l,U_sum,S_sum,B,Uxy,C):
        B1xy=[0,0]
        B2xy=[0,l]
        B3xy=[l,0]
        B4xy=[l,l]
        Ux=[i[0] for i in Uxy]
        Uy=[i[1] for i in Uxy]
        U_connect=[U_sum for i in range(U_sum)]    #初始化连接基站，编号为(U_sum)

        UBxy=Uxy+[B1xy,B2xy,B3xy,B4xy]



        def PP_down():
            p_down=[[i] for i in range(U_sum)]
            for i in range(U_sum):
                o=i
                while(1):
                    p_down[i].append(U_connect[o])
                    o=U_connect[o]
                    if o==U_sum or o==U_sum+1 or o==U_sum+2 or o==U_sum+3:
                        break
            return p_down


        def PP_up():
            p_up=[[] for i in range(U_sum)]
            for i in range(U_sum):
                for j in range(U_sum):
                     if i in P_down[j]:
                         p_up[i].append(j)
            return p_up


        c=299792458
        f=2000000000
        a=9.6
        b=0.28
        h=2500
        los=1
        nlos=20
        v=8                                         #每个包的比特数
        throughout=[0 for i in range(U_sum)]        #每个无人机的吞吐量


        def angle(x, y):
            l = np.linalg.norm([x[0] - y[0], x[1] - y[1]])
            Angle = atan(h / l)
            return Angle

        def Plos(x, y):
            plos = 1 / (1 + a * exp(-b * (angle(x, y) * 180 / pi - a)))
            return plos

        def Loss1(x, y):
            loss1 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + los
            return loss1

        def Loss2(x, y):
            loss2 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + nlos
            return loss2

        def ULoss(x, y):
            uloss = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]])) + 20 * log10(f) + 20 * log10(
                4 * pi / c) + los
            return uloss

        def MLoss(x, y):
            mloss = Plos(x, y) * Loss1(x, y) + (1 - Plos(x, y)) * Loss2(x, y) - 1.85 * (angle(x,y) - 1.41)
            return mloss

        def USNR(x, y):
            usnr = 10 ** (P / 10) / 10 ** (ULoss(x, y) / 10) / 1e-9
            return usnr

        def MSNR(x, y):
            msnr = 10 ** (P / 10) / 10 ** (MLoss(x, y) / 10) / 1e-9
            return msnr

        def UADR(x, y):
            uadr = B * log2(1 + USNR(x, y))  # 无人机跳边的可到达数据量
            return uadr

        def MADR(x, y):
            adr = B * log2(1 + MSNR(x, y))  # 无人机跳边的可到达数据量
            return adr


        def R_min(x):                                               #评价函数U的分子
            hop_R=[]
            for i in P_down[x]:
                if P_down[x].index(i)==len(P_down[x])-2:
                    break
                r=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])
                hop_R.append(r)
            if P_down[x][-1]==U_sum:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B1xy))
            elif P_down[x][-1]==U_sum+1:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B2xy))
            elif P_down[x][-1]==U_sum+2:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B3xy))
            elif P_down[x][-1]==U_sum+3:
                hop_R.append(MADR(Uxy[P_down[x][len(P_down[x])-2]],B4xy))
            r=min(hop_R)
            return r


        def delay_cost(x):                                          #绝对评价函数U的分母
            r=0
            for i in P_down[x]:
                fai=0
                for j in P_up[i]:
                    fai+=C[j]
                if P_down[x].index(i)==len(P_down[x])-2:
                    if P_down[x][-1] == U_sum:
                        u=MADR(Uxy[i],B1xy)/v
                    elif P_down[x][-1] == U_sum+1:
                        u=MADR(Uxy[i],B2xy)/v
                    elif P_down[x][-1] == U_sum+2:
                        u=MADR(Uxy[i],B3xy)/v
                    elif P_down[x][-1] == U_sum + 3:
                        u = MADR(Uxy[i], B4xy) / v
                    if u<fai:
                        return float("inf")
                    r += fai / (2 * u * (u - fai)) + 1 / u
                    break
                else:
                    u=UADR(Uxy[i],Uxy[P_down[x][P_down[x].index(i)+1]])/v
                    if u<fai:
                        return float("inf")
                r+=fai/(2*u*(u-fai))+1/u
            return r


        def U(x):
            u=R_min(x)**0.5/delay_cost(x)**0.3
            return u


        deallist=[]


        while(1):
            for i in range(U_sum):
                for j in range(U_sum+S_sum):
                    P_down = PP_down()
                    P_up = PP_up()
                    if i!=j and j not in P_up[i]:
                        begin_U_connect=U_connect[i]
                        U_begin = U(i)
                        U_connect[i] = j
                        P_down = PP_down()
                        P_up=PP_up()
                        if U(i)>U_begin:
                            continue
                        else:
                            U_connect[i]=begin_U_connect
                            continue
                if U_connect[i]==U_sum and MADR(Uxy[i],UBxy[U_connect[i]])/v<C[i]:
                    throughout[i]=MADR(Uxy[i],UBxy[U_connect[i]])/v
                else:
                    throughout[i]=C[i]

            if U_connect in deallist:
                if U_connect==deallist[-1]:
                    print('纳什收敛')
                else:
                    print('陷入循环')
                break
            deallist.append([i for i in U_connect])

        for i in range(U_sum):
            if U_connect[i]==U_sum and MADR(Uxy[i],UBxy[U_connect[i]])/v<C[i]:
                for j in range(U_sum+1, U_sum + S_sum):
                    begin_connect = U_connect[i]
                    begin_madr = MADR(Uxy[i], UBxy[U_connect[i]])
                    U_connect[i] = j
                    if MADR(Uxy[i], UBxy[U_connect[i]]) > begin_madr:
                        continue
                    else:
                        U_connect[i] = begin_connect
                throughout[i]=MADR(Uxy[i],UBxy[U_connect[i]])/v




        # print(f"总数据传输速率为:{sum(throughout)}")
        # print(throughout)

        # mp.figure(2)
        # mp.xlim(0,l)
        # mp.ylim(0,l)
        # mp.plot(Ux,Uy,'^',color='red',markersize=10,label='无人机')
        # mp.plot(B1xy[0],B1xy[1],'o',color='black',markersize=10,label='基站')
        # mp.plot(B2xy[0],B2xy[1],'o',color='black',markersize=10)
        # mp.plot(B3xy[0],B3xy[1],'o',color='black',markersize=10)
        # mp.plot(B4xy[0],B4xy[1],'o',color='black',markersize=10)
        # BUX=[i[0] for i in UBxy]
        # BUY=[i[1] for i in UBxy]
        # for i in range(U_sum):
        #     mp.arrow(BUX[i],BUY[i],BUX[U_connect[i]]-BUX[i],BUY[U_connect[i]]-BUY[i],head_length=100,width=5,length_includes_head=True,fc='blue',ec='blue',overhang=0.5)
        #
        # mp.legend()
        return throughout





    ######################################################################################################################################################################
    ######################################################################################################################################################################
    ######################################################################################################################################################################





    def backhaul_normal(l,U_sum,S_sum,B,Uxy,C):
        B1xy=[0,0]
        B2xy=[0,l]
        B3xy=[l,0]
        B4xy=[l,l]
        Ux=[i[0] for i in Uxy]
        Uy=[i[1] for i in Uxy]
        U_connect=[U_sum for i in range(U_sum)]    #初始化连接基站，编号为(U_sum)

        UBxy=Uxy+[B1xy,B2xy,B3xy,B4xy]


        c=299792458
        f=2000000000
        a=9.6
        b=0.28
        h=2500
        los=1
        nlos=20
        v=8                                         #每个包的比特数
        throughout=[0 for i in range(U_sum)]        #每个无人机的吞吐量


        def angle(x, y):
            l = np.linalg.norm([x[0] - y[0], x[1] - y[1]])
            Angle = atan(h / l)
            return Angle

        def Plos(x, y):
            plos = 1 / (1 + a * exp(-b * (angle(x, y) * 180 / pi - a)))
            return plos

        def Loss1(x, y):
            loss1 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + los
            return loss1

        def Loss2(x, y):
            loss2 = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]]) / cos(angle(x, y))) + 20 * log10(
                f) + 20 * log10(4 * pi / c) + nlos
            return loss2

        def ULoss(x, y):
            uloss = 20 * log10(np.linalg.norm([x[0] - y[0], x[1] - y[1]])) + 20 * log10(f) + 20 * log10(
                4 * pi / c) + los
            return uloss

        def MLoss(x, y):
            mloss = Plos(x, y) * Loss1(x, y) + (1 - Plos(x, y)) * Loss2(x, y) - 1.85 * (angle(x,y) - 1.41)
            return mloss

        def USNR(x, y):
            usnr = 10 ** (P / 10) / 10 ** (ULoss(x, y) / 10) / 1e-9
            return usnr

        def MSNR(x, y):
            msnr = 10 ** (P / 10) / 10 ** (MLoss(x, y) / 10) / 1e-9
            return msnr

        def UADR(x, y):
            uadr = B * log2(1 + USNR(x, y))  # 无人机跳边的可到达数据量
            return uadr

        def MADR(x, y):
            adr = B * log2(1 + MSNR(x, y))  # 无人机跳边的可到达数据量
            return adr


        for i in range(U_sum):
            for j in range(U_sum,U_sum+S_sum):
                begin_connect=U_connect[i]
                begin_madr=MADR(Uxy[i],UBxy[U_connect[i]])
                U_connect[i]=j
                if MADR(Uxy[i],UBxy[U_connect[i]])>begin_madr:
                    continue
                else:
                    U_connect[i]=begin_connect
            if MADR(Uxy[i],UBxy[U_connect[i]])/v < C[i]:
                throughout[i] = MADR(Uxy[i], UBxy[U_connect[i]]) / v
            else:
                throughout[i]=C[i]


        # print(f"总数据传输速率为:{sum(throughout)}")
        # print(throughout)

        # mp.figure(1)
        # mp.xlim(0,l)
        # mp.ylim(0,l)
        # mp.plot(Ux,Uy,'^',color='red',markersize=10,label='无人机')
        # mp.plot(B1xy[0],B1xy[1],'o',color='black',markersize=10,label='基站')
        # mp.plot(B2xy[0],B2xy[1],'o',color='black',markersize=10)
        # mp.plot(B3xy[0],B3xy[1],'o',color='black',markersize=10)
        # mp.plot(B4xy[0],B4xy[1],'o',color='black',markersize=10)
        # BUX=[i[0] for i in UBxy]
        # BUY=[i[1] for i in UBxy]
        # for i in range(U_sum):
        #     mp.arrow(BUX[i],BUY[i],BUX[U_connect[i]]-BUX[i],BUY[U_connect[i]]-BUY[i],head_length=100,width=5,length_includes_head=True,fc='blue',ec='blue',overhang=0.5)
        #
        # mp.legend()
        return throughout




    ######################################################################################################################################################################
    ######################################################################################################################################################################
    ######################################################################################################################################################################





    def random_distribution(l,U_sum):                                                   #随机分布，通用比较有代表性
        Uxy=[[randrange(0,l),randrange(0,l)] for i in range(U_sum)]
        return Uxy


    # def uniform_distribution(l,U_sum):                                                  #均匀分布，分布图和链路图不怎么好看
    #     Uxy=[]
    #     temp=sqrt(U_sum)
    #     temp=int(temp)
    #     temp2=U_sum-temp**2
    #     x=np.linspace(0+l/temp/2,l-l/temp/2,temp)
    #     for i in x:
    #         for j in x:
    #             Uxy.append([int(i),int(j)])
    #     for i in range(temp2):
    #         Uxy.append([randrange(0,l),randrange(0,l)])
    #     return Uxy


    def possion(r,lamda):                                                               #泊松分布的概率求解
        jiechen=1
        for i in range(1,r+1):
            jiechen=jiechen*i
        p=e**(-lamda)*lamda**r/jiechen
        return p
    def poisson_distribution(l):                                                        #λ线性变化的泊松分布,返回两个值，一个是Uxy的分布，一个是Uxy
        global true_number
        U_sum=0
        Uxy=[]
        lamda=0
        lamda_index=0
        while(lamda_index<l):
            lamda+=1
            lamda_index+=500
            p_list=[]
            for i in range(10):
                p_list.append(possion(i,lamda))
            temp1=uniform(0,sum(p_list))
            temp2=0
            for i in range(10):
                temp2+=p_list[i]
                if temp1<=temp2:
                    true_number = i
                    U_sum+=true_number
                    break
            temp3=0
            while(1):
                if temp3==true_number:
                    break
                x=randrange(0,lamda_index)
                y=randrange(0,lamda_index)
                if lamda_index-500<sqrt(x**2+y**2)<lamda_index:
                    Uxy.append([x,y])
                    temp3+=1
        return Uxy,U_sum



    def random_average():                                                          #多次随机分布，取平均值
        average_list=[]
        normal_R = 0
        remake1_R = 0
        remake1_R_recycle = 0
        remake2_R = 0
        for i in range(av_times):
            Uxy=Uxy_list2[i]
            normal_R += sum(backhaul_normal(l, U_sum, S_sum, B, Uxy, C))
            remake1_R += sum(backhaul_remake1(l, U_sum, S_sum, B, Uxy, C))
            remake1_R_recycle += sum(backhaul_remake1_recycle(l, U_sum, S_sum, B, Uxy, C))
            remake2_R += sum(backhaul_remake2(l, U_sum, S_sum, B, Uxy, C))
        average_R1=normal_R/av_times
        average_R2=remake1_R/av_times
        average_R2_recycle=remake1_R_recycle/av_times
        average_R3=remake2_R/av_times
        average_list.append(average_R1)
        average_list.append(average_R2)
        average_list.append(average_R2_recycle)
        average_list.append(average_R3)
        return average_list

    # def possion_average(times):                                                          #多次泊松分布，取平均值
    #     average_list=[]
    #     normal_R = 0
    #     remake1_R = 0
    #     remake2_R = 0
    #     for i in range(times):
    #         Uxy,U_sum = poisson_distribution(l)
    #         normal_R += sum(backhaul_normal(l, U_sum, S_sum, B, Uxy, C))
    #         remake1_R += sum(backhaul_remake1(l, U_sum, S_sum, B, Uxy, C))
    #         remake2_R += sum(backhaul_remake2(l, U_sum, S_sum, B, Uxy, C))
    #     average_R1=normal_R/times
    #     average_R2=remake1_R/times
    #     average_R3=remake2_R/times
    #     average_list.append(average_R1)
    #     average_list.append(average_R2)
    #     average_list.append(average_R3)
    #     return average_list

    l=12000
    S_sum=1
    B=40e+6
    P = 20
    # C=[4e+6 for i in range(U_sum)]
    C=[0.5e+6 for i in range(100)]
    U_sum=5
    x_times=6
    av_times=10                                                             #平均次数
    Uxy_list = [random_distribution(l, 100) for i in range(av_times)]

    mp.figure(dpi=100)
    x=[]
    y1=[]
    y2=[]
    y2_recycle=[]
    y3=[]
    for i in range(x_times):
        x.append(U_sum)
        Uxy_list2=[i[0:U_sum] for i in Uxy_list]
        temp=random_average()                                  #取av_times次平均
        y1.append(temp[0]*8/1000000/U_sum)
        y2.append(temp[1]*8/1000000/U_sum)
        y2_recycle.append(temp[2]*8/1000000/U_sum)
        y3.append(temp[3]*8/1000000/U_sum)
        U_sum += 5
    mp.plot(x,y1,'o:',color='black',label='直接连接基站',linewidth=1)
    mp.plot(x,y2,'+-.',color='black',label='纯策略纳什均衡方案',linewidth=1)
    mp.plot(x,y2_recycle,'x--',color='black',label='具有资源重利用过程的纯策略纳什均衡方案',linewidth=1)
    mp.plot(x,y3,'s-',color='black',label='具有资源重利用的改进策略纳什均衡方案',linewidth=1)
    mp.xlim((5,30))
    mp.ylim((0,4))
    mp.xticks(np.arange(5,35,5))
    mp.yticks(np.arange(0,4.5,0.5))
    mp.legend()
    mp.xlabel('N/个')
    mp.ylabel('平均每个UAV的有效回程传输速率/Mbps')
    # backhaul_normal(l,U_sum,S_sum,B,Uxy,C)
    # backhaul_remake1(l,U_sum,S_sum,B,Uxy,C)
    # backhaul_remake2(l,U_sum,S_sum,B,Uxy,C)
    mp.savefig("..\..\仿真报告\批注修改图-衰落\无人机数目变量-平均有效回程速率.png", dpi=750, bbox_inches = 'tight')

mp.show()
