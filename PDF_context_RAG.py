import sys
from os import getenv
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,QTextEdit, QFileDialog, QLabel, QLineEdit, QListWidget, QMessageBox
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

openai.api_key = getenv("OPENAI_API_KEY")

class DocQAApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Document Q&A using RAG")
        self.resize(1000, 600)


        main_layout = QHBoxLayout()
        self.setLayout(main_layout)


        self.chat_history = QListWidget()
        main_layout.addWidget(self.chat_history, 30)


        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 70)

        self.label = QLabel("add PDFs:")
        right_panel.addWidget(self.label)

        self.upload_btn = QPushButton("Upload PDFs")
        self.upload_btn.clicked.connect(self.load_pdfs)
        right_panel.addWidget(self.upload_btn)

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Ask a question...")
        right_panel.addWidget(self.question_input)

        self.ask_btn = QPushButton("Get Answer")
        self.ask_btn.clicked.connect(self.answer_question)
        right_panel.addWidget(self.ask_btn)

        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        right_panel.addWidget(self.answer_box)


        self.chunks = []
        self.index = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_pdfs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open PDFs", "", "PDF Files (*.pdf)")
        if files:
            try:
                combined_text = ""
                for file_path in files:
                    pdf_reader = PdfReader(file_path)
                    for page in pdf_reader.pages:
                        if page.extract_text():
                            combined_text += page.extract_text()
                self.chunks = [combined_text[i:i+500] for i in range(0, len(combined_text), 500)]
                embeddings = self.model.encode(self.chunks)
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(np.array(embeddings))
                QMessageBox.information(self, "Success", f"{len(files)} PDFs loaded and indexed!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PDFs: {e}")

    def answer_question(self):
        if not self.chunks or self.index is None:
            QMessageBox.warning(self, "Warning", "Please upload PDFs first.")
            return

        question = self.question_input.text()
        if not question:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
            return

        try:
            q_embed = self.model.encode([question])
            D, I = self.index.search(np.array(q_embed), k=3)
            retrieved_chunks = [self.chunks[i] for i in I[0]]

            prompt = f"Answer the question using only the context below:\n\n{retrieved_chunks}\n\nQuestion: {question}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}]
            )
            answer = response["choices"][0]["message"]["content"]

            
            self.chat_history.addItem(f"Q: {question}\nA: {answer}\n{'-'*40}")
            self.answer_box.setText(answer)
            self.question_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate answer: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DocQAApp()
    window.show()
    sys.exit(app.exec())
