import numpy as np
import pandas as pd
import tqdm

def coordinateConcateVelocity(track_arr, frame_speed):
    num_row = len(track_arr)
    unit_time = (1/frame_speed) * 1000
    vel_arr = np.full((num_row, 5), 0, dtype=np.float64)
    for i in range(len(track_arr)): # x,y좌표 → x,y 속도계산
        #좌표 변환
        if (i < num_row - 1):
            if (track_arr[i,0] == track_arr[i+1, 0]):
                vel_arr[i,0]= track_arr[i,0]
                
                frame_diff = track_arr[i+1,3] - track_arr[i,3]
                vel_arr[i,1]= (track_arr[i+1,1]- track_arr[i,1]) / (unit_time*frame_diff)
                vel_arr[i,2]= (track_arr[i+1,2] - track_arr[i,2]) / (unit_time*frame_diff)
                
                vel_arr[i,3] = np.sqrt(vel_arr[i,1]**2 + vel_arr[i,2]**2)
                
                '''temp_degree = np.degrees(np.arctan(np.abs(vel_arr[i,2])/vel_arr[i,1]))
                vel_arr[i,4] = temp_degree if temp_degree > 0 else temp_degree + 180'''
                vel_arr[i,4] = 0
            
            elif (i == num_row):
                vel_arr[i,0] = vel_arr[i-1,0]
            else:
                vel_arr[i,0] = vel_arr[i-1,0]
    
    track_arr = np.delete(track_arr, [5,6,7,10], axis=1)
    track_arr = np.hstack((track_arr, vel_arr[:,1:])) #속도(x,y),전체속도, 각도 추가
    
    return track_arr

#%% calculate xvelocity from intensity information

def calculate_xvelo(inten_arr, laser_length=0.005):
    # inten_arr [:,0]:Frame [:,1]:Max intensity [:,2]:Sum of intensity
    
    final_frame = np.max(inten_arr[:,0])
    start_frame = np.min(inten_arr[:,0])
    diff_frame = final_frame - start_frame + 1
    
    ''' laser length = 3cm = 0.03 m
    intensity 값이 상위 30%이내인 프레임들 중 가장 큰 프레임 : max_frame
    intensity 값이 상위 30%이내인 프레임들 중 가장 작은 프레임 : min_frame
    프레임 개수 = max_frame - min_frame + 1 '''
    
    '''intenMax_col = inten_arr[:,8]
    threshold = np.percentile(intenMax_col, 40, method='nearest')
    inlaser_rows = inten_arr[intenMax_col >= threshold]
    
    max_frame = np.max(inlaser_rows[:,3])
    min_frame = np.min(inlaser_rows[:,3])''' 
    #보류. 프레임 수가 작아서 그 안에서도 자르면 속도가 너무 커짐 
    
    time = diff_frame * 0.0002
    xvelocity = laser_length / time
    if (xvelocity < 6):
        return xvelocity
    else:
        xvelocity = 6
        return xvelocity
#%%
def preprocess_intensity(file_path, arr, frame_speed=5000):
    origin_x = arr[0]
    origin_y = arr[1]
    data = pd.read_csv(file_path,low_memory=False)
    data = data.iloc[4:]
    df = pd.DataFrame(data)
    df_intenstiy = df.iloc[:,12:18].applymap(float)
    df = df.iloc[:,2:10]
    df_r = df.drop(df.columns[[1,4,6]], axis=1).applymap(float)

    df_final = pd.concat([df_r, df_intenstiy], axis=1)

    track_arr = df_final.sort_values([df_final.columns[0], df_final.columns[3]],ascending=True).to_numpy()

    del df_r, df, df_intenstiy
    del df_final, data
        
    '''열에 들어있는 0:Track_ID [1,2]:position_x,y 3:position_T 4:radius 5: inten_mean
    6: inten_median 7:inten_min 8: inten_max 9 : inten_total 10:inten_std'''

    num_rows = len(track_arr)
    vel_arr = np.full((num_rows, 5), 0, dtype=np.float64)
    
    unit_time = (1/frame_speed) * 1000 # 1000 : mm → m 변환 
    
    track_arr[:,1] -= origin_x
    track_arr[:,2] -= origin_y
            
    # 1. raw data - 속도 데이터 병합 (concade)
    #track_arr = np.hstack((track_arr, vel_arr[:,1:])) #속도(x,y),전체속도, 각도 추가
    
    track_arr = coordinateConcateVelocity(track_arr, frame_speed=5000)
    '''[0] : track_id / [1]:x coord / [2]:y coordinate / [3] : frame_nubmer /[4]:Radius(mm) 
    [5]:intensity_max / [6]:intensity_sum / [7]: x velocity /[8]: y velocity [9] total velocity 
    [10] velocity angle'''
    
    # 2.track_id의 고유값 추출 -> id_arr
    id_arr = np.unique(track_arr[:, 0]).reshape(-1,1)
    track_one = np.full((len(id_arr), 9), 0 ,dtype=np.float32)
    splash_list = []
    
    # 전체 데이터에서 track_id의 고유값과 일치하는 행들만 모아서 처리 (selected_rows)
    # 1. 벽에 맞고 튀므로 y값이 작아져야함. 커지는 애들은 주입하는 액적이므로 제외 
    #(Front에서는 틀림)
    # 첫 프레임의 y 좌표가 마지막 프레임의 y 좌표보다 큰 track_id만 splash_list에 넣음. 
    #(Front에서는 틀림)
    
    # 첫 프레임과 중위 각도,속도를 따로 추출하여 frame_angle에 저장함.
    
    step = 0
    for index, track_id in enumerate(id_arr):
        selected_rows = track_arr[track_arr[:, 0] == track_id, :]
        
        
        min_frame = np.min(selected_rows[:,3])
        track_one[index,0] = min_frame #track_one 0번째 열: 시작 프레임
            
        initial_x = selected_rows[0, 1] #제일 처음 값들 = 초기값
        initial_y = selected_rows[0, 2]
        track_one[index,1] = initial_x #초기 x좌표
        track_one[index,2] = initial_y #초기 y좌표
        
        median_velocity = np.median(selected_rows[:,9])
        track_one[index,3] = median_velocity #median velocity (total)
        
        median_angle = np.median(selected_rows[:,10])
        track_one[index,4] = median_angle # median angle
    
        median_radius = np.percentile(selected_rows[:,4],5, method='nearest') 
        track_one[index,5] = median_radius # 5% (small) Radius (mm)
        
        median_xvelocity = np.median(selected_rows[:,7])
        track_one[index,6] = median_xvelocity
        
        median_yvelocity = np.median(selected_rows[:,8])
        track_one[index,7] = median_yvelocity
            
        # x velocity 계산을 위한 슬라이싱 -> intensity_arr
        intensity_arr = selected_rows[:,[3,5,6]]
        xvelocity = calculate_xvelo(intensity_arr)
        track_one[index, 8] = xvelocity
        
        step += 1
        print('\r{:6.2f}'.format((step)/len(id_arr)*100), end='\r')
        
    '''[0]:id [1]:start frame [2]:initial x [3]:initial y [4]: total velocity 
    [5]: velocity angle [6]:diameter [7]:x velocity [8]:y velocity (median)'''
    
    '''02.26 각도값 계산 안하고 0으로 통일함.'''
    
    id_arr = np.hstack((id_arr, track_one))
    #splash_arr = id_arr[np.isin(id_arr[:,0], splash_list)]
    
    #return splash_arr
    print("done\n")
    return id_arr