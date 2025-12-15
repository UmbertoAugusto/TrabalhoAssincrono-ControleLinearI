#############################################################################
#
# Universidade Federal do Rio de Janeiro
# Centro de Tecnologia - Escola Politecnica
# Departamento de Engenharia Eletronica e de Computacao
# Disciplina: Controle Linear I
# Professor: Heraldo Almeida
# Aluno: Umberto Augusto de A. P. Pereira
# DRE: 123170891
#
# Trabalho Assincrono (TA) - 2025/2
#
#############################################################################

import numpy as np
import sympy as sp
import control as ct
import matplotlib.pyplot as plt

partes = '123'

# digitos do DRE (123170891)
d1 = 1
d2 = 9
d3 = 8

# Parametros da Tabela
V  = 10            # Tensão simétrica nos potenciômetros
n  = 10            # Número de voltas dos potenciômetros
Ka = 150           # Ganho do amplificador de potência
a  = 150           # Frequência de corte do amplificador de potência
Ra = 4 + d1        # Resistência da armadura
Ja = 0.02          # Inércia rotacional da armadura
Da = 0.01          # Atrito viscoso da armadura
Kb = 0.5           # Constante de força contraeletromotriz
Kt = 0.5           # Constante de torque do motor
N1 = 25            # Dentes na engrenagem do eixo do motor
N2 = 250           # Dentes na engrenagem do eixo da antena
N3 = 250           # Dentes na engrenagem do potenciômetro
JL = 1 + d2        # Inércia rotacional da carga/antena
DL = 1 + d3        # Atrito viscoso da carga/antena

# Parametros calculados a partir dos anteriores
Kpot = V / (n * np.pi)                  # Constante do Potenciômetro
Jm = Ja + ( (N1/N2)**2 * JL )           # Inércia Equivalente na Armadura
Dm = Da + ( (N1/N2)**2 * DL )           # Atrito Viscoso Equivalente na Armadura
Km = Kt / (Ra * Jm)                     # Constante da Tranferencia do Motor
am = (1.0/Jm) * (Dm + (Kt * Kb / Ra))   # Polo da Tranferencia do Motor
Kg = N1 / N2                            # Theta_o / Theta_m = N1 / N2

# Definindo as Funcoes de Transferencia de cada bloco
s = sp.symbols('s')
K = sp.symbols('K', real=True, positive=True) # ganho ajustavel do root locus

input_pot = Kpot                # transdutor entrada
G_pre_amp = K                   # pre amplificador
G_power_amp = Ka / (s + a)      # amplificador de potencia
G_motor = Km / (s * (s + am))   # motor
G_gears_forward = N1 / N2       # engrenagens caminho direto
H_gears_feedback = N2 / N3      # engrenagens realimentacao
H_output_pot = Kpot             # transdutor saida 

# -----------------------------------------------------------
# PARTE 1
# -----------------------------------------------------------
if '1' in partes:
    # -------------------------------------------------------------------------------------------------------------------------------
    # Questao 1

    FT_malha_aberta = input_pot * G_pre_amp * G_power_amp * G_motor * G_gears_forward
    print("\n--- 1b. Função de Transferência Malha Aberta (Numérica) ---")
    sp.pprint(sp.simplify(FT_malha_aberta))

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 2

    numerador, denominador = sp.fraction(sp.simplify(FT_malha_aberta))
    polos = sp.solve(denominador, s)

    print("\n--- 2. Polos de Malha Aberta (em rad/s) ---")
    print(f"Polos encontrados: {polos}")

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 3

    Theta_i_s = 1/s #degrau
    Theta_o_s = FT_malha_aberta * Theta_i_s

    print("\n--- 3. Expressão da Saída no Domínio da Frequência Theta_o(s) ---")
    sp.pprint(Theta_o_s)

    # passando para o dominio do tempo
    theta_o_t = sp.inverse_laplace_transform(Theta_o_s, s, sp.symbols('t'))
    print("\n--- 3. Expressão no Tempo theta_o(t) ---")
    sp.pprint(theta_o_t)

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 4

    # substitui K=10 e cria sistema usando biblioteca control para plotar resposta ao degrau
    FT_num_sympy = FT_malha_aberta.subs(K, 10)
    num_poly = sp.Poly(sp.numer(FT_num_sympy), s).all_coeffs() # numerador do polinomio
    den_poly = sp.Poly(sp.denom(FT_num_sympy), s).all_coeffs() # denominador do polinomio

    # listas de floats representando os coeficientes
    num_float = [float(x) for x in num_poly] 
    den_float = [float(x) for x in den_poly]
    sys_open_loop = ct.tf(num_float, den_float) # cria o sistema como objeto da biblioteca control

    print(f"\nSistema Malha Aberta (Control): \n{sys_open_loop}")

    # plota resposta ao degrau
    T_sim, y_sim = ct.step_response(sys_open_loop, T=10)

    plt.figure()
    plt.plot(T_sim, y_sim)
    plt.title('Item 4: Resposta ao Degrau em Malha Aberta (K=10)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo de Saída (rad)')
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------
# PARTE 2
# -----------------------------------------------------------
if '2' in partes:
    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 5

    G = G_pre_amp * G_power_amp * G_motor * G_gears_forward # caminho direto
    H = H_gears_feedback * H_output_pot # realimentacao
    FT_malha_fechada = input_pot * ( G / (1 + G * H) ) # funcao de transferencia do sistema em malha fechada
    
    print("\n--- 5b. Função de Transferência de Malha Fechada ---")
    FT_MF_Final = sp.simplify(FT_malha_fechada)
    sp.pprint(FT_MF_Final)

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 6
    print("\n--- 6. Cálculos Auxiliares do Lugar das Raízes ---")

    print("\n--- 6e. Pontos de Partida/Chegada (Breakaway/Break-in) ---")
    G_sem_K = G_power_amp * G_motor * G_gears_forward
    GH = G_sem_K * H # calcula GH deixando o ganho K de fora
    # calcula A, B, A' e B'
    num, den = sp.fraction(sp.simplify(GH))
    derivada_num = sp.diff(num, s)
    derivada_den = sp.diff(den, s)
    # encontra valores de s que resolvem A'B-AB' = 0
    expressao = num*derivada_den - den*derivada_num
    candidatos_s = sp.solve(expressao, s)
    print(f"Possíveis valores de s: {candidatos_s}")
    # filtra valores possiveis de s, sabendo que ha ponto de partida entre os polos de menor valor
    for valor_s in candidatos_s:
        if valor_s < 0 and valor_s > -1.25:
            ponto_partida = valor_s
    # calcula valor de K que leva ao polo duplo no ponto de partida
    valor_GH = GH.subs(s, ponto_partida)
    # 1+kGH = 0 -> k = -1/GH
    val_K = -1/valor_GH
    print(f"Ponto de partida s = {valor_s} com ganho K = {val_K:.4f}")

    print("\n--- 6f. Interseção com o Eixo Imaginário ---")
    # 1 + kGH = 0 -> denominador_GH + k*numerador_GH
    eq = den + K*num
    # substitui s por j*omerga
    omega = sp.symbols('omega', real=True, positive=True)
    eq_jw = eq.subs(s, sp.I*omega)
    # separa partes real e imaginaria da equacao
    eq_real = sp.re(eq_jw)
    eq_imag = sp.im(eq_jw)
    # encontra valores de omega usando parte imaginaria
    solucoes_omega = sp.solve(eq_imag, omega)
    print(f"Frequências de cruzamento (omega) encontradas: {solucoes_omega}")
    # encontra valores de K para cada omega encontrado
    for w_val in solucoes_omega:
        k_sol = sp.solve(eq_real.subs(omega, w_val), K)
        if len(k_sol) > 0:
            print(f"  -> Para omega = {w_val:.4f} rad/s, K crítico = {k_sol[0]:.4f}")


    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 7
    print("\n--- 7. Gerando o Lugar das Raízes ---")
    GH_sympy = sp.simplify(GH)
    print("G(s)H(s):")
    sp.pprint(GH_sympy)
    # Convertendo Sympy -> Control
    # polinomios
    num_poly_gh = sp.Poly(sp.numer(GH_sympy), s).all_coeffs()
    den_poly_gh = sp.Poly(sp.denom(GH_sympy), s).all_coeffs()
    # converte para float
    num_float_gh = [float(x) for x in num_poly_gh]
    den_float_gh = [float(x) for x in den_poly_gh]
    #gera objeto da biblioteca control
    sys_gh = ct.tf(num_float_gh, den_float_gh)
    print(f"\nFunção de Laço GH(s) (Control Library):\n{sys_gh}")
    # plotando root locus
    plt.figure()
    dados_rlocus = ct.root_locus(sys_gh, grid=True)
    plt.title('Item 7: Lugar das Raízes (Root Locus)')
    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.show()

# -----------------------------------------------------------
# PARTE 3
# -----------------------------------------------------------
if '3' in partes:

    G = G_pre_amp * G_power_amp * G_motor * G_gears_forward # caminho direto
    G_sem_K = G_power_amp * G_motor * G_gears_forward
    H = H_gears_feedback * H_output_pot # realimentacao
    FT_malha_fechada = sp.simplify(input_pot * ( G / (1 + G * H) ))

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 8

    FT_malha_fechada_K_substituido = FT_malha_fechada.subs(K, 14.6649) # substitui valor de K escolhido
    print("\n--- 8. Função de Transferência de Malha Fechada com K substituido---")
    sp.pprint(FT_malha_fechada_K_substituido)
    
    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 9

    ts = 4 / 0.6237 # calcula valor do tempo de assentamento de acordo com o K escolhido
    print(f"\n--- 9. Tempo de assentamento ts = {ts:.4f} ---")

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 10
    print(f"\n--- 10. Cálculo do erro Estácionário para Entrada Degrau ---")

    R_s = 1/s # entrada degrau

    E_s = R_s / (1 + (G * H).subs(K, 14.6649)) # erro estacionario para o K definido
    e_t = sp.limit(s * E_s, s, 0) # calculando no tempo
    print(f"Sinal Erro Estacionário (rad) = {float(e_t)}")

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 11
    print(f"\n--- 11. Simulação da Resposta ao Degrau do Sistema ---")

    # convertendo para sistema da bilbioteca control
    num_poly = sp.Poly(sp.numer(FT_malha_fechada_K_substituido), s).all_coeffs() # numerador do polinomio
    den_poly = sp.Poly(sp.denom(FT_malha_fechada_K_substituido), s).all_coeffs() # denominador do polinomio

    # listas de floats representando os coeficientes
    num_float = [float(x) for x in num_poly] 
    den_float = [float(x) for x in den_poly]
    sys_critic = ct.tf(num_float, den_float) # cria o sistema como objeto da biblioteca control
    print(f"\nSistema Malha Fechada (Control): \n{sys_critic}")

    # plota resposta ao degrau
    T_sim, y_sim = ct.step_response(sys_critic, T=15)

    plt.figure()
    plt.plot(T_sim, y_sim, 'b', label='Resposta do Sistema')
    plt.axhline(y=0.98, color='r', linestyle='--', label='Limite 2% Inferior') # para visualizar tolerancia de 2%
    plt.title('Item 11: Resposta ao Degrau em Malha Fechada (K=14.6649)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo de Saída (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 12

    R_s = 1/(s**2) # entrada rampa

    E_s = R_s / (1 + (G * H).subs(K, 14.6649)) # erro estacionario para o K definido
    e_t = float(sp.limit(s * E_s, s, 0))
    print(f"\n--- 12. Erro Estacionário (rad) para entrada Rampa = {e_t:.4f} ---")
    print(f"    (Em graus: {np.degrees(e_t):.4f}°)")

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 13

    t_sim = np.linspace(0, 60, 1000) # simula por 60 segundos

    r_t = t_sim # sinal de entrada, rampa

    T_out, y_out = ct.forced_response(sys_critic, T=t_sim, U=r_t) # pega resposta do sistema para a entrada rampa
    erro_t = r_t - y_out  # calcula erroa theta_i - theta_o

    plt.figure()
    plt.plot(t_sim, erro_t, label='Erro de Rastreamento (Rampa)')
    # plota o erro estacionario teorico para comparacao
    plt.axhline(y=e_t, color='r', linestyle='--', label=f'Erro Estacionário Teórico ({e_t:.4f})')
    plt.title('Item 13: Erro para Entrada Rampa (Malha Fechada)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # pega o erro do tempo maximo calculado (60s)
    print(f"Erro final para a entrada rampa na simulação: {erro_t[-1]:.4f} rad")

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 14
    print(f"\n--- 14. Projeto de Controlador Lead Lag ---")

    G_controlador_lead_sem_K = (s+am) / (s+50) # parte Lead do controlador
    # usa-se am (=1.25) para eviter diferencas por erro numerico

    # Conferindo se ponto de partida está a esquerda de -6.5
    print(f"\nCalculando ponto partida do eixo real e K que gera polo duplo nesse ponto")
    # calcula A, B, A' e B'
    GH_lead_semK = G_controlador_lead_sem_K * G_sem_K * H
    num, den = sp.fraction(sp.simplify(GH_lead_semK))
    derivada_num = sp.diff(num, s)
    derivada_den = sp.diff(den, s)

    # encontra valores de s que resolvem A'B-AB' = 0
    expressao = num*derivada_den - den*derivada_num
    candidatos_s = sp.solve(expressao, s)
    print(f"Possíveis valores de s: {candidatos_s}")

    # filtra valores possiveis de s, sabendo que ha ponto de partida entre os polos de menor valor
    for valor_s in candidatos_s:
        if valor_s < 0 and valor_s > -50: # o polo em -1,25 foi cancelado pelo zero no mesmo valor
            ponto_partida = valor_s

    # calcula valor de K que leva ao polo duplo no ponto de partida
    valor_GH_semK = GH_lead_semK.subs(s, ponto_partida)
    valor_K = -1/valor_GH_semK # 1+kGH = 0 -> k = -1/GH
    print(f"Ponto de partida s = {valor_s:.4f} com ganho K = {valor_K:.4f}")


    # Calculando K necessario para reduzir o tempo de assentamento em 10 vezes
    eq_caracteristica = 1 + K * G_controlador_lead_sem_K * G_sem_K * H # 1 + KG(s)H(s) = 0
    expressao_final = eq_caracteristica.subs(s, -6.5) # -6.5 foi o valor calculado
    ganho = sp.solve(expressao_final, K)
    valor_K = float(ganho[0])
    print(f"\nValor do Ganho que garante um polo em -6.5: {valor_K}")


    # Projeto da parte Lag

    # Calculando erro estacionario sem parte Lag
    expressao_Kv = s * valor_K * G_controlador_lead_sem_K * G_sem_K * H
    Kv = float(sp.limit(expressao_Kv, s, 0))
    print(f"\nConstante de Velocidade sem inclusao da parte Lag: Kv = {Kv:.4f}")

    # Implementando controlador Lead Lag
    G_control_LeadLag = valor_K * ((s + 0.06) / (s + 0.001)) * G_controlador_lead_sem_K
    print(f"\nExpressao do Controlador Lead Lag:")
    sp.pprint(G_control_LeadLag)

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 15
    print(f"\n--- 15. Simulacoes com Controlador Lead Lag  ---")

    FT_MalhaFechada_ControladorLeadLag = sp.simplify(Kpot * (G_control_LeadLag * G_sem_K) / (1 + G_control_LeadLag * G_sem_K * H))
    print("\nFuncao de Transferencia de Malha Fechada com Controlador Lead Lag:")
    sp.pprint(FT_MalhaFechada_ControladorLeadLag)

    # convertendo para sistema da bilbioteca control
    num_poly = sp.Poly(sp.numer(FT_MalhaFechada_ControladorLeadLag), s).all_coeffs() # numerador do polinomio
    den_poly = sp.Poly(sp.denom(FT_MalhaFechada_ControladorLeadLag), s).all_coeffs() # denominador do polinomio

    # listas de floats representando os coeficientes
    num_float = [float(x) for x in num_poly]
    den_float = [float(x) for x in den_poly]
    sys_LeadLag = ct.tf(num_float, den_float) # cria o sistema como objeto da biblioteca control
    print(f"\nSistema Malha Fechada com Controlador Lead Lag (Control): \n{sys_LeadLag}")

    t_sim = np.linspace(0, 150, 1000) # simula ate 150 segundos

    r_t = t_sim # sinal de entrada, rampa

    T_out, y_out = ct.forced_response(sys_LeadLag, T=t_sim, U=r_t) # pega resposta do sistema para a entrada rampa
    erro_t = r_t - y_out  # calcula erroa theta_i - theta_o

    plt.figure()
    plt.plot(t_sim, erro_t, label='Erro de Rastreamento (Rampa)')
    # plota o erro estacionario desejado para comparacao
    plt.axhline(y=0.0031214, color='r', linestyle='--', label=f'Erro Estacionário Teórico ({0.0031214})')
    plt.title('Item 15a: Erro para Entrada Rampa Com Controlador Lead Lag')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # pega o erro do tempo maximo calculado (150s)
    print(f"\nErro final para a entrada rampa na simulação: {erro_t[-1]:.4f} rad")

    # plota resposta ao degrau
    T_sim, y_sim = ct.step_response(sys_LeadLag, T=15)

    plt.figure()
    plt.plot(T_sim, y_sim, 'b', label='Resposta do Sistema')
    plt.axhline(y=0.98, color='r', linestyle='--', label='Limite 2% Inferior') # para visualizar tolerancia de 2%
    plt.axhline(y=1.02, color='g', linestyle='--', label='Limite 2% Superior') # para visualizar tolerancia de 2%
    plt.title('Item 15b: Resposta ao Degrau em Malha Fechada com Controlador Lead Lag')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo de Saída (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 16
    print(f"\n--- 16. Projeto de Controlador PID ---")

    G_controlador_PD_sem_K = (s+am) # parte derivativa do controlador
    # usa-se am (=1.25) para eviter diferencas por erro numerico

    # Conferindo se ponto de partida está a esquerda de -6.5
    print(f"\nCalculando ponto partida do eixo real e K que gera polo duplo nesse ponto")
    # calcula A, B, A' e B'
    GH_PD_semK = G_controlador_PD_sem_K * G_sem_K * H
    num, den = sp.fraction(sp.simplify(GH_PD_semK))
    derivada_num = sp.diff(num, s)
    derivada_den = sp.diff(den, s)

    # encontra valores de s que resolvem A'B-AB' = 0
    expressao = num*derivada_den - den*derivada_num
    candidatos_s = sp.solve(expressao, s)
    print(f"Possíveis valores de s: {candidatos_s}")

    # filtra valores possiveis de s, sabendo que ha ponto de partida entre os polos de menor valor
    for valor_s in candidatos_s:
        if valor_s < 0 and valor_s > -150: # o polo em -1,25 foi cancelado pelo zero no mesmo valor
            ponto_partida = valor_s

    # calcula valor de K que leva ao polo duplo no ponto de partida
    valor_GH_semK = GH_PD_semK.subs(s, ponto_partida)
    valor_K = -1/valor_GH_semK # 1+kGH = 0 -> k = -1/GH
    print(f"Ponto de partida s = {ponto_partida:.4f} com ganho K = {valor_K:.4f}")


    # Calculando K necessario para reduzir o tempo de assentamento em 10 vezes
    eq_caracteristica = 1 + K * G_controlador_PD_sem_K * G_sem_K * H # 1 + KG(s)H(s) = 0
    expressao_final = eq_caracteristica.subs(s, -6.5) # -6.5 foi o valor calculado
    ganho = sp.solve(expressao_final, K)
    valor_K = float(ganho[0])
    print(f"\nValor do Ganho que garante um polo em -6.5: {valor_K:.4f}")


    # Projeto da parte PI
    # Controlador PI vai zerar erro estacionario para entrada rampa
    G_control_PID = valor_K * ((s + 0.1) / s) * G_controlador_PD_sem_K
    print(f"\nExpressao do Controlador PID:")
    sp.pprint(G_control_PID)

    # -------------------------------------------------------------------------------------------------------------------------------
    print("\n")
    # Questao 17
    print(f"\n--- 17. Simulacoes com Controlador PID  ---")

    FT_MalhaFechada_ControladorPID = sp.simplify(Kpot * (G_control_PID * G_sem_K) / (1 + G_control_PID * G_sem_K * H))
    print("\nFuncao de Transferencia de Malha Fechada com Controlador PID:")
    sp.pprint(FT_MalhaFechada_ControladorPID)

    # convertendo para sistema da bilbioteca control
    num_poly = sp.Poly(sp.numer(FT_MalhaFechada_ControladorPID), s).all_coeffs() # numerador do polinomio
    den_poly = sp.Poly(sp.denom(FT_MalhaFechada_ControladorPID), s).all_coeffs() # denominador do polinomio

    # listas de floats representando os coeficientes
    num_float = [float(x) for x in num_poly]
    den_float = [float(x) for x in den_poly]
    sys_PID = ct.tf(num_float, den_float) # cria o sistema como objeto da biblioteca control
    print(f"\nSistema Malha Fechada com Controlador PID (Control): \n{sys_PID}")

    t_sim = np.linspace(0, 150, 1000) # simula ate 150 segundos

    r_t = t_sim # sinal de entrada, rampa

    T_out, y_out = ct.forced_response(sys_PID, T=t_sim, U=r_t) # pega resposta do sistema para a entrada rampa
    erro_t = r_t - y_out  # calcula erroa theta_i - theta_o

    plt.figure()
    plt.plot(t_sim, erro_t, label='Erro de Rastreamento (Rampa)')
    # plota o erro estacionario desejado para comparacao
    plt.axhline(y=0, color='r', linestyle='--', label=f'Erro Estacionário Teórico ({0.0})')
    plt.title('Item 17a: Erro para Entrada Rampa Com Controlador PID')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # pega o erro do tempo maximo calculado (150s)
    print(f"\nErro final para a entrada rampa na simulação: {erro_t[-1]:.4f} rad")

    # plota resposta ao degrau
    T_sim, y_sim = ct.step_response(sys_PID, T=15)

    plt.figure()
    plt.plot(T_sim, y_sim, 'b', label='Resposta do Sistema')
    plt.axhline(y=0.98, color='r', linestyle='--', label='Limite 2% Inferior') # para visualizar tolerancia de 2%
    plt.axhline(y=1.02, color='g', linestyle='--', label='Limite 2% Superior') # para visualizar tolerancia de 2%
    plt.title('Item 17b: Resposta ao Degrau em Malha Fechada com Controlador PID')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo de Saída (rad)')
    plt.grid(True)
    plt.legend()
    plt.show()