boolean issuccess = false;
boolean cur = true;

int success = 0;
int blink;
int psec;
int timestamp;
int normt0;
int counter = 0;
int cond_idx = (int)random(8);
int num_cond = 0;
boolean keyrelease = false;
boolean[] cond_visit = { false, false, false, false, false, false, false, false };

void draw(){
  randomSeed(millis());
  background(#000000);
  if(issuccess){
    if(millis() - blink >= 200) issuccess = false;
    fill(#00FF00);
  }else fill(#320000);
  rect(400, 50, 100, 300);
  fill(#FFFFFF);

  if(counter < 35){
    cond_visit[cond_idx] = true;
    if(cur) ellipse(conds[cond_idx].xpos, 200, 40, 40);
    conds[cond_idx].move();
    if(millis() - psec >= conds[cond_idx].ct){
      TableRow nRow = table.addRow();
      nRow.setInt("timestamp", timestamp);
      nRow.setInt("cond", cond_idx + 1);
      nRow.setInt("trial", counter + 1);
      nRow.setInt("success", success);
      nRow.setInt("t_cue", conds[cond_idx].tcue);
      nRow.setInt("t_zone", conds[cond_idx].tzone);
      nRow.setInt("p", conds[cond_idx].ct);
      nRow.setInt("key", keyrelease ? 0 : 1);
      timestamp = success = 0;
      cur = true;
      counter++;
      conds[cond_idx].setpos();
      normt0 = psec = millis();
      normt0 += conds[cond_idx].tcue;
    }
  }else{
    counter = 0;
    num_cond++;
    switch (num_cond) {
      case 8 :
        keyrelease = true;
        for(int i = 0; i < 8; i++)cond_visit[i] = false;
        cond_idx = (int)random(8);
        break;
      case 16 :
        saveTable(table, "2018147558.csv");
        exit();
        break;
    }
    if(num_cond < 16)
      while(cond_visit[cond_idx])cond_idx = (int)random(8);
  }
}

void keyPressed(){
  if(!keyrelease && cur){
    if(key == ' '){
      if(millis() - normt0 >= 0 && millis() - normt0 <= conds[cond_idx].tzone){
        issuccess = true;
        success = 1;
        blink = millis();
      }
      timestamp = millis() - normt0;
      cur = false;
    }
  }
}

void keyReleased(){
  if(keyrelease && cur){
    if(key == ' '){
      if(millis() - normt0 >= 0 && millis() - normt0 <= conds[cond_idx].tzone){
        issuccess = true;
        success = 1;
        blink = millis();
      }
      timestamp = millis() - normt0;
      cur = false;
    }
  }
}