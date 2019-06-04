function simplest_neural_network

    training_input = [
    0 0 1,
    0 1 0,
    0 1 1,
    1 0 0,
    1 1 0, 
    1 0 1];

    traing_result = [
    0, 
    0, 
    0,
    1,
    1,
    1
    ];

    weights = [
    rand , rand , rand 
    ];

    biasias = [
    rand, rand, rand    
    ];

%     disp(weights );
%     disp(biasias );
    
    function T = sigmoid(x)
        T = 1 / (1 + exp(-x));
    end
    
    function [weight_update, biasias_update, loss] = calculate_updates(inputs, outputs, weights, biasias)
        input_samples_count = size(training_input,1);
        weight_updates = zeros(input_samples_count, 3);
        biasias_updates = zeros(input_samples_count, 3);
        loss = zeros(input_samples_count, 3);
        
          for t=1:1:(input_samples_count -1)
            plain_results = inputs(t, :).*weights+biasias;
            normalized_result = sigmoid(sum(plain_results));
            loss = (outputs(t) - normalized_result)^2;
%             disp(loss);
            derivate_of_loss = 2*(outputs(t) - normalized_result);
            derivative_of_normalized_result = (exp(-t))./((1+exp(-t)).^2);

            for weight_instance=1:1:3
                derivative_loss_weight = derivate_of_loss * inputs(t, weight_instance) * derivative_of_normalized_result;
                weight_updates(t, weight_instance) =  derivative_loss_weight;
            end

            derivative_loss_bias = derivate_of_loss * derivative_of_normalized_result;
             for weight_instance=1:1:3
                 biasias_updates(t,weight_instance) =  derivative_loss_bias;
                 %partial_derivative_loss_bias = derivate_of_loss * derivate_of_bias * derivative_of_normalized_result;
             end
        end
        
        weight_update = (sum(weight_updates)/3);
        biasias_update = (sum(biasias_updates)/3);
    end


    for i=1:1:500
        [weight_update, biasias_update, loss] = calculate_updates(training_input, traing_result, weights, biasias);
        weights = weights + weight_update;
        biasias = biasias + biasias_update;      
%         disp(loss)
    end    
    
%     disp(weights);
%     disp(biasias);
     
    testing_input = [
    1, 1, 1;
    0, 0, 0
    ];

    testing_output = [
    1,
    0
    ];
end