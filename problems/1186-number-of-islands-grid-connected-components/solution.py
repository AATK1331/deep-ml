import numpy as np
def count_islands(grid):
    if not grid:
        return 0
    A=np.array(grid)
    m,n=A.shape
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    count=0
    for i in range(m):
        for j in range(n):
            if A[i][j]==1:
                count+=1
                q=[(i,j)]
                A[i][j]=0
                while q:
                    x,y=q.pop(0)
                    for dx,dy in dirs:
                        nx=x+dx
                        ny=y+dy
                        if 0<=nx<m and 0<=ny<n and A[nx][ny]==1:
                            A[nx][ny]=0
                            q.append((nx,ny))
    return  count
                    
