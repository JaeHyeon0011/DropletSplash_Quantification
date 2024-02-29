import numpy as np
import pandas as pd

def Preprocess(file_path, origin_x, origin_y, frame_speed):
    data = pd.read_csv(file_path,low_memory=False)
    data = data.iloc[4:]
    df = pd.DataFrame(data)
    df = df.iloc[:,2:10]
    df_r = df.drop(df.columns[[1,4,6]], axis=1).applymap(float)
    #df_r = df_r.applymap(float)
    
    #track 갯수 구하기, 트랙번호, 좌표, 직경, 시간(프레임) 추출
    df_sort = df_r.sort_values([df_r.columns[0], df_r.columns[3]],ascending=True)
        
    track_arr = df_sort.to_numpy()
    num_rows = len(track_arr)
    vel_arr = np.full((num_rows, 5), 0, dtype=np.float64)
    
    unit_time = (1/frame_speed) * 1000 # 1000 : mm → m 변환 
    
    track_arr[:,1] -= origin_x
    track_arr[:,2] -= origin_y
    
    #좌표 데이터에서 속도 데이터 계산
    for i in range(len(track_arr)):
        #좌표 변환
        if (i < num_rows - 1):
            if (track_arr[i,0] == track_arr[i+1, 0]):
                vel_arr[i,0]= track_arr[i,0]
                
                frame_diff = track_arr[i+1,3] - track_arr[i,3]
                vel_arr[i,1]= (track_arr[i+1,1]- track_arr[i,1]) / (unit_time*frame_diff)
                vel_arr[i,2]= (track_arr[i+1,2] - track_arr[i,2]) / (unit_time*frame_diff)
                
                vel_arr[i,3] = np.sqrt(vel_arr[i,1]**2 + vel_arr[i,2]**2)
                
                temp_degree = np.degrees(np.arctan(np.abs(vel_arr[i,2])/vel_arr[i,1]))
                vel_arr[i,4] = temp_degree if temp_degree > 0 else temp_degree + 180
            
            elif (i == num_rows):
                vel_arr[i,0] = vel_arr[i-1,0]
            else:
                vel_arr[i,0] = vel_arr[i-1,0]
                
    # 1.track_id의 고유값 추출 -> id_arr
    # 2. raw data - 속도 데이터 병합 (concade)
    id_arr = np.unique(track_arr[:, 0]).reshape(-1,1)
    track_arr = np.hstack((track_arr, vel_arr[:,1:])) #속도(x,y),전체속도, 각도 추가
    #return track_arr
    
    track_one = np.full((len(id_arr), 8), 0 ,dtype=np.float32)
    splash_list = []
    
    # 전체 데이터에서 track_id의 고유값과 일치하는 행들만 모아서 처리
    # 1. 벽에 맞고 튀므로 y값이 작아져야함. 커지는 애들은 주입하는 액적이므로 제외
    # 첫 프레임의 y 좌표가 마지막 프레임의 y 좌표보다 큰 track_id만 splash_list에 넣음.
    # 첫 프레임과 중위 각도,속도를 따로 추출하여 frame_angle에 저장함.
    for index, track_id in enumerate(id_arr):
        selected_rows = track_arr[track_arr[:, 0] == track_id, :]
        min_index = np.argmin(selected_rows[:,3])
        max_index = np.argmax(selected_rows[:,3])
        
        if selected_rows[min_index,2] >= selected_rows[max_index,2]:
            splash_list.append(track_id)
            
            min_frame = np.min(selected_rows[:,3])
            track_one[index,0] = min_frame #track_one 0번째 열: 시작 프레임
            
            initial_x = selected_rows[min_index, 1]
            initial_y = selected_rows[min_index, 2]
            track_one[index,1] = initial_x #초기 x좌표
            track_one[index,2] = initial_y #초기 y좌표
            
            median_velocity = np.median(selected_rows[:,7])
            track_one[index,3] = median_velocity #median velocity (total)
            
            median_angle = np.median(selected_rows[:,-1])
            track_one[index,4] = median_angle # median angle
        
            median_diameter = np.percentile(selected_rows[:,4],90) 
            track_one[index,5] = median_diameter # median Diameter
            
            median_xvelocity = np.median(selected_rows[:,5])
            track_one[index,6] = median_xvelocity
            
            median_yvelocity = np.median(selected_rows[:,6])
            track_one[index,7] = median_yvelocity
            
    
    id_arr = np.hstack((id_arr, track_one))
    splash_arr = id_arr[np.isin(id_arr[:,0], splash_list)]
    
    return splash_arr
