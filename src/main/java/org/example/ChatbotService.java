package org.example;

import jakarta.annotation.PostConstruct;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

@Service
public class ChatbotService {

    private final VectorStore vectorStore;
    private final ChatClient chatClient;

    public ChatbotService(
            VectorStore vectorStore,
            ChatClient.Builder chatClientBuilder
    ) {
        this.vectorStore = vectorStore;
        this.chatClient = chatClientBuilder.build();
    }

    // Load the knowledge base when the application starts
    @PostConstruct
    public void loadKnowledgeBase() {

        try {

            // 1. Read all PDF files from the knowledge directory
            PathMatchingResourcePatternResolver resolver =
                    new PathMatchingResourcePatternResolver();

            Resource[] pdfFiles = resolver.getResources(
                    "classpath:knowledge/*.pdf"
            );

            if (pdfFiles.length == 0) {
                System.out.println(
                        "No PDF files found in knowledge directory."
                );
                return;
            }

            System.out.println(
                    "Found " + pdfFiles.length + " PDF files."
            );

            // 2. Initialize the text splitter
            TokenTextSplitter splitter =
                    TokenTextSplitter.builder()
                            .withChunkSize(800)
                            .withMinChunkSizeChars(350)
                            .withMinChunkLengthToEmbed(5)
                            .withMaxNumChunks(10000)
                            .withKeepSeparator(true)
                            .build();

            // 3. Process each PDF
            for (Resource pdfFile : pdfFiles) {

                String fileName = pdfFile.getFilename();

                System.out.println(
                        "Processing PDF: " + fileName
                );

                // Read PDF
                PagePdfDocumentReader pdfReader =
                        new PagePdfDocumentReader(pdfFile);

                List<Document> documents = pdfReader.read();

                System.out.println(
                        "Extracted " + documents.size()
                                + " documents from " + fileName
                );

                // 4. Divide documents into chunks
                List<Document> chunks =
                        splitter.apply(documents);

                System.out.println(
                        "Generated " + chunks.size()
                                + " chunks from " + fileName
                );

                // 5. Store chunks in Pinecone
                // Spring AI generates embeddings internally
                vectorStore.add(chunks);

                System.out.println(
                        "Successfully stored chunks in Pinecone: "
                                + fileName
                );
            }

            System.out.println(
                    "Knowledge base loading completed successfully."
            );

        } catch (IOException e) {

            throw new RuntimeException(
                    "Failed to load knowledge base PDFs.", e
            );

        } catch (Exception e) {

            throw new RuntimeException(
                    "Error while processing knowledge base.", e
            );
        }
    }

    // Answer the user's question using RAG
    public String answerUserQuery(String question) {

        // 1. Question -> Vector
        // 2. Similarity search inside our vector DB
        // 3. Fetch top 4 relevant chunks

        List<Document> relevantChunks =
                vectorStore.similaritySearch(
                        SearchRequest.builder()
                                .query(question)
                                .topK(4)
                                .build()
                );

        // Return fallback if no relevant information is found
        if (relevantChunks.isEmpty()) {
            return "I don't have that information in the company documents.";
        }

        // 4. Build context from retrieved documents
        StringBuilder context = new StringBuilder();

        for (Document document : relevantChunks) {

            context.append(document.getText())
                    .append("\n\n");
        }

        String finalContext = context.toString();

        // 5. Create the system prompt
        String prompt = """
                You are an AI customer support assistant for our e-commerce company.

                Answer the customer using ONLY the company information provided below.

                If the answer is not available in the provided information, say:
                "I don't have that information in the company documents."

                COMPANY INFORMATION:
                %s
                """.formatted(finalContext);

        // 6. Send system prompt and user query to the LLM
        return chatClient.prompt()
                .system(prompt)
                .user(question)
                .call()
                .content();
    }
}