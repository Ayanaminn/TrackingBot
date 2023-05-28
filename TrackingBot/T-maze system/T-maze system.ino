/*
Interface wtih Python through COM port communication
Read input
*/

String userCommand = "";
int ledpin1 = 13;
int ledpin2 = 11;
int ledpin3 = 10;
int ledpin4 = 7;
int ledpin5 = 6;
int ledpin6 = 5;

// negative to pin2
// positive to pin 3
int directionPin = 12;//Motor forward or backward
int pwmPin = 3; //Motor speed
int brakePin = 9;//Motor brake

bool motionState; //Switch on or off state

// the setup function runs once when you press reset or power the board
void setup() {
  Serial.begin(19200);
  // initialize digital pin 9 as an output.
  pinMode(ledpin1, OUTPUT);
  pinMode(ledpin2, OUTPUT);
  pinMode(ledpin3, OUTPUT);
  pinMode(ledpin4, OUTPUT);
  pinMode(ledpin5, OUTPUT);
  pinMode(ledpin6, OUTPUT);
  // initial state
//  analogWrite (ledpin,0);
  digitalWrite(ledpin1,LOW);
  digitalWrite(ledpin2,LOW);
  digitalWrite(ledpin3,LOW);
  digitalWrite(ledpin4,LOW);
  digitalWrite(ledpin5,LOW);
  digitalWrite(ledpin6,LOW);
  
// for default LED connection test  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

// Motors
  //Setup Channel A
  pinMode(directionPin, OUTPUT); //Initiates Motor Channel A pin
  pinMode(brakePin, OUTPUT); //Initiates Brake Channel A pin
  pinMode(pwmPin, OUTPUT);
  motionState == false;
}

// the loop function runs over and over again forever
void loop() {
  while (Serial.available())
  {  
////////////////////////////////////////  
    userCommand = Serial.readStringUntil('\n');
    if (userCommand == "01"){ 
      digitalWrite(LED_BUILTIN, HIGH); 
      digitalWrite(ledpin1,HIGH);
    }
    if (userCommand == "00"){
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(ledpin1,LOW);
    }
    if (userCommand == "11"){
      digitalWrite(ledpin2,HIGH);
    }
    if (userCommand == "10"){
      digitalWrite(ledpin2,LOW);
    }      
    if (userCommand == "21"){
       digitalWrite(ledpin3,HIGH);
    }       
    if (userCommand == "20"){
      digitalWrite(ledpin3,LOW);
    }      
    if (userCommand == "31"){
      digitalWrite(ledpin4,HIGH);
    }  
    if (userCommand == "30"){
      digitalWrite(ledpin4,LOW);
    } 
    if (userCommand == "41"){
       digitalWrite(ledpin5,HIGH);
    }   
    if (userCommand == "40"){
      digitalWrite(ledpin5,LOW);
    }  
    if (userCommand == "51"){
       digitalWrite(ledpin6,HIGH);
    }  
    if (userCommand == "50"){
      digitalWrite(ledpin6,LOW);
    } 
    if (userCommand == "61"){

       if(motionState == false){
              digitalWrite(directionPin, HIGH); //High goes forward
                //release breaks
              digitalWrite(brakePin, LOW);
              
              //set work duty for the motor
              analogWrite(pwmPin, 100);
            
              //Make sure motor complete action
              delay(1500);
            
              //activate breaks
              digitalWrite(brakePin, HIGH);
              // and set work duty for the motor to 0 (off)
              analogWrite(pwmPin, 0);
              
              delay(1000);
              motionState = true;
            }
          else{
              //  
              //  delay(2000);
          }
    }
    
    if (userCommand == "60"){

       if(motionState == true){
          digitalWrite(directionPin, LOW); //LOW goes backward
            //release breaks
          digitalWrite(brakePin, LOW);
          
          //set work duty for the motor
          analogWrite(pwmPin, 100);
        
          //Make sure motor complete action
          delay(1500);
        
          //activate breaks
          digitalWrite(brakePin, HIGH);
          // and set work duty for the motor to 0 (off)
          analogWrite(pwmPin, 0);
          
          delay(1000);
          motionState = false;
        }
      
        else{
            //  
            //  delay(2000);
        }
    }
  }
}
