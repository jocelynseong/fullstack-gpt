from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import random

# OpenAI 모델 초기화
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,  # GPT 모델 응답의 랜덤성을 최소화
)

# 랜덤 3자리 숫자 생성 (중복 없음)
def generate_random_number():
    digits = list(range(1, 10))  # 1-9까지 숫자
    random.shuffle(digits)
    return "".join(map(str, digits[:3]))

# 숫자 비교 함수
def check_guess(secret, guess):
    strikes = sum(1 for a, b in zip(secret, guess) if a == b)
    balls = len(set(secret) & set(guess)) - strikes
    return strikes, balls

# 타겟 숫자
target_number = generate_random_number()
print(f"[DEBUG] Target Number: {target_number}")  # 디버깅용, 실제 게임에서는 주석 처리 가능

# 시스템 메시지 템플릿 생성
system_message_template = SystemMessagePromptTemplate.from_template("""
def check_guess(secret, guess):
    strikes = sum(1 for a, b in zip(secret, guess) if a == b)
    balls = len(set(secret) & set(guess)) - strikes
    return strikes, balls
                                                                    
You are an assistant for a Baseball Game. Your role is to explain the results of the user's guess.

Rules:
1. The secret number is always a unique 3-digit number from 1 to 9 with no duplicates.
2. The user will guess a 3-digit number. Compare it with the secret number using the function below:
   - A "strike" means a correct digit in the correct position.
   - A "ball" means a correct digit but in the wrong position.
3. For example:
   - Secret = "123", Guess = "132" → Result = 1 Strike, 2 Balls
   - Secret = "456", Guess = "789" → Result = 0 Strikes, 0 Balls
4. If the user gets "3 Strikes", they win.

Your task:
1. Receive the user's guess as input.
2. Use the result from `check_guess(secret, guess)` to provide feedback.
3. Format the result as: "N Strikes, M Balls".
4. If "3 Strikes", congratulate the user and declare them the winner.
""")


# 게임 진행 함수
def play_game():
    print("Welcome to the Baseball Game!")
    print("Try to guess the 3-digit number. Enter 'quit' to exit the game.")
    print("")
    
    while True:
        # 사용자 입력
        user_guess = input("Your guess (3 digits): ").strip()

        if user_guess.lower() == "quit":
            print("Thanks for playing! Goodbye!")
            break

        # 유효성 검사
        if len(user_guess) != 3 or not user_guess.isdigit() or len(set(user_guess)) != 3 or user_guess[0] == "0":
            print("Invalid input. Please enter a unique 3-digit number starting from 1-9.")
            continue

        # Python에서 숫자 비교
        strikes, balls = check_guess(target_number, user_guess)

        # LangChain LLM 호출
        system_message_content = system_message_template.format()
        system_message = SystemMessage(content=str(system_message_content))
        human_message = HumanMessage(content=f"The user guessed {user_guess}. The result is {strikes} Strikes, {balls} Balls.")
        
        response = llm.predict_messages([
            system_message,
            human_message
        ])

        # AI 응답 출력
        print(f"AI: {response.content}")

        # 종료 조건 확인
        if strikes == 3:
            print("Congratulations! You guessed the correct number.")
            break

# 게임 실행
play_game()
