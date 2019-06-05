function simplest_neural_network

    training_input = [
    0 0 0 1,
    0 0 1 0,
    0 0 1 1,
    0 1 0 0,
    0 1 0 1, 
    0 1 1 0,
    0 1 1 1,
    1 0 0 0,
    ];

    traing_result = [
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
    ];

    weights = [
    rand , rand , rand , rand
    ];

    biasias = [
    rand, rand, rand , rand  
    ];

%     disp(weights );
%     disp(biasias );
    
    function T = sigmoid(x)
        T = 1 / (1 + exp(-x));
    end
    
    function [weight_update, biasias_update, loss] = calculate_updates(inputs, outputs, weights, biasias)
        input_samples_count = size(inputs,1);
        weight_updates = zeros(input_samples_count, 4);
        biasias_updates = zeros(input_samples_count, 4);
        loss = zeros(input_samples_count, 1);
        
          for t=1:1:(input_samples_count)
            plain_results = inputs(t, :).*weights+biasias;
            
            normalized_result = sigmoid(sum(plain_results));
%             loss(t) = (outputs(t) - normalized_result)^2;
            predicted_output = normalized_result >= 0.5;
            loss(t) = sqrt((outputs(t) - predicted_output).^2);
%             disp(loss);
            derivate_of_loss = 2*(outputs(t) - normalized_result);
            derivative_of_normalized_result = (exp(-t))./((1+exp(-t)).^2);

            for weight_instance=1:1:4
                derivative_loss_weight = derivate_of_loss * inputs(t, weight_instance) * derivative_of_normalized_result;
                weight_updates(t, weight_instance) =  derivative_loss_weight;
            end

            derivative_loss_bias = derivate_of_loss * derivative_of_normalized_result;
             for weight_instance=1:1:4
                 biasias_updates(t,weight_instance) =  derivative_loss_bias;
                 %partial_derivative_loss_bias = derivate_of_loss * derivate_of_bias * derivative_of_normalized_result;
             end
        end
        
        weight_update = (sum(weight_updates)/4);
        biasias_update = (sum(biasias_updates)/4);
    end

    loss_to_be_aware_of = 0.75;
    while loss_to_be_aware_of > 0.125 
        [weight_update, biasias_update, loss] = calculate_updates(training_input, traing_result, weights, biasias);
        weights = weights + weight_update;
        biasias = biasias + biasias_update;   
        loss_to_be_aware_of = sum(loss)/8;
        disp(loss_to_be_aware_of);
%         disp(loss)
    end   
%       disp(loss)
      disp(weights);
      disp(biasias);
     
    testing_input = [
    1 0 0 1;
    1 0 1 0;
    1 0 1 1;
    1 1 0 0;
    ];

    testing_output = [
    1,
    0,
    1,
    0
    ];



    [weight_update, biasias_update, loss] = calculate_updates(testing_input, testing_output, weights, biasias);
    disp(sum(loss)/4);    
end