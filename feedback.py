from datetime import datetime
from pathlib import Path
import json

ARQUIVO_PADRAO = Path("feedback.log")

def salvar_feedback(usuario: str, mensagem: str, arquivo: Path = ARQUIVO_PADRAO, avaliacao: str = None) -> None:
    """
    Função para salvar os feedbacks em formato JSON por linha

    Args:
        usuario: Nome do usuário
        mensagem: Texto do feedback
        arquivo: Caminho do arquivo para salvar
        avaliacao: Nota da avaliação

    Return:
        None
    """
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "usuario": usuario,
        "mensagem": mensagem,
        "avaliacao": avaliacao
    }

    with arquivo.open("a", encoding="utf-8") as file:
        file.write(json.dumps(feedback, ensure_ascii=False) + "\n")


def salvar_pergunta(pergunta):

    if pergunta:
        with open("history.log", "a", encoding="utf-8") as file:
            file.write(f"{pergunta}\n")
